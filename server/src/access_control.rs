use std::{
    collections::HashSet,
    fs::{self, read},
    path::{Path, PathBuf},
};

use ring::digest::{digest, SHA256};
use tonic::Status;
use x509_parser::prelude::Pem;

/// Rename for [`Vec<u8>`].
pub type PubKey = Vec<u8>;

/// This struct holds the hash of the PEM formatted public keys loaded from the directory
/// passed to the `KeyManagement` struct at start-up.
///
/// If the directory doesn't contain either of these directories (`owners`, `users`), then it fails to start
/// All public keys in `owners` are stored in `KeyManagement::owners` and those for users are
/// stored in `KeyManagement::owners`
#[derive(Debug, Default, Clone)]
pub struct KeyManagement {
    owners: HashSet<PubKey>,
    users: HashSet<PubKey>,
}

impl KeyManagement {
    fn read_from_file(path: PathBuf) -> Result<PubKey, Status> {
        let file = &read(path)?;
        let (_, Pem { contents, .. }) = x509_parser::pem::parse_x509_pem(&file[..])
            .map_err(|e| Status::aborted(e.to_string()))?;

        let hash = hex::encode(digest(&SHA256, &contents[..]));
        Ok(hash.as_bytes().to_vec())
    }

    pub fn load_from_dir(path: String) -> Result<Self, Status> {
        let mut owners: HashSet<PubKey> = HashSet::new();
        let mut users: HashSet<PubKey> = HashSet::new();

        // Check for this directory structure
        // -- keys
        // -----| owners
        // -----| users

        if !Path::new(&path).is_dir() {
            Err(Status::aborted("Please provide a directory!"))?
        }

        // Contains sub-directories.
        let paths = fs::read_dir(path.clone())?;

        for path in paths {
            let path = path?;
            if path.path().is_dir() {
                let files = path.path().read_dir()?;
                match path.path().file_name() {
                    Some(v) => {
                        let dir = v.to_str().ok_or_else(|| Status::aborted(""))?;
                        match dir {
                            "owners" => {
                                for file in files {
                                    let file = file?;
                                    owners.insert(KeyManagement::read_from_file(file.path())?);
                                }
                            }
                            "users" => {
                                for file in files {
                                    let file = file?;
                                    users.insert(KeyManagement::read_from_file(file.path())?);
                                }
                            }
                            _ => (),
                        }
                    }
                    None => {
                        return Err(Status::internal(
                            "owners and users public keys not provided!",
                        ))?;
                    }
                }
            }
        }

        if owners.is_empty() && users.is_empty() {
            Err(Status::aborted(format!(
                "Please provided these directories [`owners`, `users`] in {}",
                path
            )))?
        }

        Ok(KeyManagement { owners, users })
    }

    pub fn verify_key(&self, pub_key: &str) -> Result<(), Status> {
        let key = pub_key.as_bytes().to_vec();
        if !(self.owners.contains(&key) || self.users.contains(&key)) {
            Err(Status::aborted(format!("{:?} not authenticated!", pub_key)))?
        }
        Ok(())
    }
}
