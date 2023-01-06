use std::{collections::HashMap, fs, path::Path};

use crate::prelude::*;
use ring::{
    digest::{digest, SHA256},
    signature,
};
use tonic::{metadata::MetadataMap, Status};
use x509_parser::prelude::Pem;

pub type PubKey = Vec<u8>;

#[derive(Debug, Default, Clone)]
pub struct KeyManagement {
    owners: HashMap<String, PubKey>,
    users: HashMap<String, PubKey>,
}

impl KeyManagement {
    fn read_from_file(path: &Path) -> Result<(String, PubKey)> {
        let file = &fs::read(path).with_context(|| anyhow!("Reading file: {path:?}"))?;
        let (_, Pem { contents, .. }) = x509_parser::pem::parse_x509_pem(&file[..])
            .with_context(|| anyhow!("Parsing PEM file: {path:?}"))?;

        let hash = hex::encode(digest(&SHA256, &contents[..]));
        Ok((hash, contents))
    }

    pub fn get_hash_and_keys(dir: fs::ReadDir) -> Result<HashMap<String, PubKey>> {
        let mut res = HashMap::new();
        for entry in dir {
            let entry = entry.context("Listing files in directory")?;
            let file = entry.path();

            let (hash, raw) = Self::read_from_file(&file)
                .with_context(|| anyhow!("Reading file: {:?}", &file))?;
            res.insert(hash, raw);
        }
        Ok(res)
    }

    pub fn load_from_dir(path: &Path) -> Result<Self, Status> {
        if !Path::new(&path).is_dir() {
            Err(Status::aborted("Please provide a public keys directory!"))?
        }
        let owners_path = &path.join("owners");
        let owners =
            fs::read_dir(owners_path).map_err(|_| Status::aborted("No owners directory found!"))?;

        let users_path = &path.join("users");
        let users =
            fs::read_dir(users_path).map_err(|_| Status::aborted("No users directory found!"))?;

        let owners = KeyManagement::get_hash_and_keys(owners)
            .map_err(|_| Status::aborted("There is an issue with the owner's key!"))?;

        let users = KeyManagement::get_hash_and_keys(users)
            .map_err(|_| Status::aborted("There is an issue with the user's key!"))?;

        Ok(Self { owners, users })
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
                let keys = &mut self.owners.iter().chain(self.users.iter());

                if let Some((_, raw_pub)) = keys.find(|&(k, _v)| public_key_hash.to_string().eq(k))
                {
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
                return Err(Status::permission_denied(format!(
                    "{:?} not authenticated!",
                    public_key_hash
                )));
            }
            None => Err(Status::permission_denied(format!(
                "No signature provided for public key {}",
                hex::encode(public_key_hash)
            ))),
        }
    }

    pub fn verify_owner(&self, public_key_hash: &str) -> bool {
        /*
            For authentication, we check if the provided public key exists in the list of owner public keys provided at start-up.
        */
        if self.owners.contains_key(public_key_hash) {
            return true;
        }
        return false;
    }
}
