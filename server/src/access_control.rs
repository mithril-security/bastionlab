use std::{
    collections::HashMap,
    fs::{self, read},
    net::SocketAddr,
    path::{Path, PathBuf},
};

use prost::Message;
use ring::{
    digest::{digest, SHA256},
    signature,
};
use tonic::{metadata::MetadataMap, Request, Status};
use x509_parser::prelude::Pem;

/// Type alias for [`Vec<u8>`].
pub type PubKey = Vec<u8>;

/// This struct holds a hashmap of hashes of the PEM formatted public keys as keys and
/// the loaded PEM formatted public key as value  loaded from the directory
/// passed to the [`KeyManagement::load_from_file`] struct at start-up.
///
/// If the directory doesn't contain either of these directories (`owners`, `users`), then it fails to start
/// All public keys in `owners` are stored in `KeyManagement::owners` and those for users are
/// stored in `KeyManagement::owners`
#[derive(Debug, Default, Clone)]
pub struct KeyManagement {
    owners: HashMap<String, PubKey>,
    users: HashMap<String, PubKey>,
}

impl KeyManagement {
    fn read_from_file(path: PathBuf) -> Result<(String, PubKey), Status> {
        let file = &read(path)?;
        let (_, Pem { contents, .. }) = x509_parser::pem::parse_x509_pem(&file[..])
            .map_err(|e| Status::aborted(e.to_string()))?;

        let hash = hex::encode(digest(&SHA256, &contents[..]));
        Ok((hash, contents))
    }

    pub fn load_from_dir(path: String) -> Result<Self, Status> {
        let mut owners: HashMap<String, PubKey> = HashMap::new();
        let users: HashMap<String, PubKey> = HashMap::new();

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
                                    let (hash, raw) = KeyManagement::read_from_file(file.path())?;
                                    owners.insert(hash, raw);
                                }
                            }
                            "users" => {
                                for file in files {
                                    let file = file?;
                                    let (hash, raw) = KeyManagement::read_from_file(file.path())?;
                                    owners.insert(hash, raw);
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

    pub fn verify_signature(
        &self,
        public_key_hash: &str,
        message: &[u8],
        header: &MetadataMap,
    ) -> Result<(), Status> {
        /*
            For authentication, first of we check if the provided public key exists in the list of public keys
            provided at start-up (owners, users).

            If it exists, we go ahead to then verify the signature received from the client by verifying the signature
            with the loaded public key created using `signature::UnparsedPublicKey::new`.
        */
        match header.get_bin(format!("signature-{}-bin", public_key_hash)) {
            Some(signature) => {
                let joined = self
                    .owners
                    .iter()
                    .chain(self.users.iter())
                    .collect::<HashMap<&String, &PubKey>>();

                if let Some(raw_pubkey) = joined.get(&public_key_hash.to_string()) {
                    let raw_pub = *raw_pubkey;
                    let public_key = spki::SubjectPublicKeyInfo::try_from(raw_pub.as_ref())
                        .map_err(|_| {
                            Status::invalid_argument(format!(
                                "Invalid SubjectPublicKeyInfo for pubkey {}",
                                public_key_hash
                            ))
                        })?;
                    let public_key = signature::UnparsedPublicKey::new(
                        &signature::ECDSA_P256_SHA256_ASN1,
                        public_key.subject_public_key,
                    );

                    let sign = signature.to_bytes().map_err(|_| {
                        Status::invalid_argument(format!(
                            "Could not decode signature for public key {}",
                            public_key_hash
                        ))
                    })?;

                    public_key.verify(message, &sign).map_err(|_| {
                        Status::permission_denied(format!(
                            "Invalid signature for public key {}",
                            public_key_hash
                        ))
                    })?;
                    return Ok(());
                }
                Err(Status::aborted(format!(
                    "{:?} not authenticated!",
                    public_key_hash
                )))?
            }
            None => Err(Status::permission_denied(format!(
                "No signature provided for public key {}",
                hex::encode(public_key_hash)
            ))),
        }
    }
}

pub fn get_message<T: Message>(method: &[u8], req: &Request<T>) -> Result<Vec<u8>, Status> {
    let meta = req
        .metadata()
        .get_bin("challenge-bin")
        .ok_or_else(|| Status::invalid_argument("No challenge in request metadata"))?;
    let challenge = meta
        .to_bytes()
        .map_err(|_| Status::invalid_argument("Could not decode challenge"))?;

    let mut res =
        Vec::with_capacity(method.len() + challenge.as_ref().len() + req.get_ref().encoded_len());
    res.extend_from_slice(method);
    res.extend_from_slice(challenge.as_ref());
    req.get_ref()
        .encode(&mut res)
        .map_err(|e| Status::internal(format!("error while encoding the request: {:?}", e)))?;
    Ok(res)
}

pub fn verify_ip(stored: &SocketAddr, recv: &SocketAddr) -> bool {
    stored.ip().eq(&recv.ip())
}
