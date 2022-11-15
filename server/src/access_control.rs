use std::{
    ffi::OsStr,
    fs::{self, read, File, ReadDir},
    path::{Path, PathBuf},
};

use tonic::Status;
pub type PubKey = Vec<u8>;
#[derive(Debug, Default, Clone)]
pub struct KeyManagement {
    owners: Vec<PubKey>,
    users: Vec<PubKey>,
}

impl KeyManagement {
    fn read_from_file(path: PathBuf) -> Result<PubKey, Status> {
        let file = read(path)?;
        Ok(file)
    }
    pub fn load_from_dir(path: String) -> Result<Self, Status> {
        let mut owners: Vec<PubKey> = Vec::new();
        let mut users: Vec<PubKey> = Vec::new();
        if Path::new(&path).is_dir() {
            // Contains sub-directories.
            let paths = fs::read_dir(path)?;
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
                                        owners.push(KeyManagement::read_from_file(file.path())?);
                                    }
                                }
                                "users" => {
                                    for file in files {
                                        let file = file?;
                                        users.push(KeyManagement::read_from_file(file.path())?);
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
        } else {
        }

        Ok(KeyManagement { owners, users })
    }
}
