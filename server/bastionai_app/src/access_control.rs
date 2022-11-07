use log::*;
use prost::Message;
use ring::{digest, signature};
use serde::{Deserialize, Serialize};
use tonic::{metadata::MetadataMap, Request, Status};

use crate::remote_torch::{self, Empty, ReferenceRequest, RunRequest, TestRequest, TrainRequest};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Rule {
    AtLeastNOf(usize, Vec<Rule>),
    WithCheckpoint(serde_bytes::ByteBuf),
    WithDataset(serde_bytes::ByteBuf),
    // IssuedBy(serde_bytes::ByteBuf),
    // HasSecret(serde_bytes::ByteBuf),
    SignedWith(serde_bytes::ByteBuf),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct License {
    pub train: Rule,
    pub train_metric: Rule,
    pub test: Rule,
    pub test_metric: Rule,
    pub list: Rule,
    pub fetch: Rule,
    pub delete: Rule,
    pub result_strategy: ResultStrategy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResultStrategy {
    Checkpoint, // Copy checkpoint's license
    Dataset,    // Copy checkpoint's license
    And,        // Use checkpoint's licence AND dataset's
    Or,         // Use checkpoint's licence OR dataset's
    Custom(Box<License>),
}

impl Rule {
    fn verify(
        &self,
        header: &MetadataMap,
        message: &[u8],
        user_public_key_hash: Option<&[u8]>,
        checkpoint_hash: Option<&[u8]>,
        dataset_hash: Option<&[u8]>,
    ) -> Result<(), Status> {
        match self {
            Rule::AtLeastNOf(n, policies) => {
                let mut m = 0;
                let mut failed = String::new();
                for (i, policy) in policies.iter().enumerate() {
                    match policy.verify(
                        header,
                        message,
                        user_public_key_hash,
                        checkpoint_hash,
                        dataset_hash,
                    ) {
                        Ok(_) => m += 1,
                        Err(e) => failed.push_str(&format!("\nRule #{}: {}", i, e.message())),
                    }
                    if m >= *n {
                        return Ok(());
                    }
                }
                Err(Status::permission_denied(&format!(
                    "Only {} subrules matched but at least {} are required.\nFailed sub rules are:",
                    m, n
                )))
            }
            Rule::SignedWith(raw_public_key) => {
                let public_key_hash = hex::encode(digest::digest(&digest::SHA256, raw_public_key));
                match header.get_bin(format!("signature-{}-bin", public_key_hash)) {
                    Some(meta) => {
                        let public_key = spki::SubjectPublicKeyInfo::try_from(
                            raw_public_key.as_ref(),
                        )
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

                        let sign = meta.to_bytes().map_err(|_| {
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
                        Ok(())
                    }
                    None => Err(Status::permission_denied(format!(
                        "No signature provided for public key {}",
                        hex::encode(public_key_hash)
                    ))),
                }
            }
            // Rule::HasSecret(secret) => {
            //     if let Some(meta) = header.get_bin("secret") {
            //         let header_secret = meta.to_bytes().map_err(|_| {
            //             Status::invalid_argument("Could not decode secret")
            //         })?;
            //         if &*secret == &*header_secret {
            //             return Ok(());
            //         }
            //     }
            //     Err(Status::permission_denied("Missing secret"))
            // }
            // Rule::IssuedBy(hash) => {
            //     if let Some(user_public_key_hash) = user_public_key_hash {
            //         if user_public_key_hash == hash {
            //             Ok(())
            //         } else {
            //             Err(Status::permission_denied("The Issuer did not provide a certificate"))
            //         }
            //     } else {
            //         Err(Status::permission_denied("Issuer mismatch"))
            //     }
            // }
            Rule::WithCheckpoint(hash) => {
                if let Some(checkpoint_hash) = checkpoint_hash {
                    if &*hash != checkpoint_hash {
                        return Err(Status::permission_denied("Checkpoint mismatch"));
                    }
                }
                Ok(())
            }
            Rule::WithDataset(hash) => {
                if let Some(dataset_hash) = dataset_hash {
                    if &*hash != dataset_hash {
                        return Err(Status::permission_denied("Dataset mismatch"));
                    }
                }
                Ok(())
            }
        }
    }
}

fn get_message<T: Message>(method: &[u8], req: &Request<T>) -> Result<Vec<u8>, Status> {
    let mut res = Vec::from(method);
    if let Some(meta) = req.metadata().get_bin("challenge") {
        let challenge = meta
            .to_bytes()
            .map_err(|_| Status::invalid_argument("Could not decode challenge"))?;
        res.append(&mut challenge.to_vec());
    }
    res.append(&mut req.get_ref().encode_to_vec());
    Ok(res)
}

impl License {
    pub fn verify_fetch(&self, req: &Request<ReferenceRequest>) -> Result<(), Status> {
        self.fetch.verify(
            &req.metadata(),
            &get_message(b"fetch", req)?,
            None,
            None,
            None,
        )
    }
    pub fn verify_list(&self, req: &Request<Empty>) -> Result<(), Status> {
        self.list.verify(
            &req.metadata(),
            &get_message(b"list", req)?,
            None,
            None,
            None,
        )
    }
    pub fn verify_delete(&self, req: &Request<ReferenceRequest>) -> Result<(), Status> {
        self.list.verify(
            &req.metadata(),
            &get_message(b"delete", req)?,
            None,
            None,
            None,
        )
    }
    pub fn verify_train(&self, req: &Request<TrainRequest>) -> Result<(), Status> {
        self.train.verify(
            &req.metadata(),
            &get_message(b"train", req)?,
            None,
            Some(&req.get_ref().model[..]),
            Some(&req.get_ref().dataset[..]),
        )
    }
    pub fn verify_test(&self, req: &Request<TestRequest>) -> Result<(), Status> {
        self.train.verify(
            &req.metadata(),
            &get_message(b"test", req)?,
            None,
            Some(&req.get_ref().model[..]),
            Some(&req.get_ref().dataset[..]),
        )
    }
    pub fn combine(&self, other: &License) -> Result<License, Status> {
        let l1 = if let ResultStrategy::Custom(l) = &self.result_strategy {
            l
        } else {
            self
        };
        let l2 = if let ResultStrategy::Custom(l) = &other.result_strategy {
            l
        } else {
            other
        };

        if l1.result_strategy != l2.result_strategy {
            return Err(Status::permission_denied(
                "Checkpoint and dataset licenses are not combatible",
            ));
        }

        Ok(match l1.result_strategy {
            ResultStrategy::Checkpoint => l1.clone(),
            ResultStrategy::Dataset => l2.clone(),
            ResultStrategy::And => License {
                train: Rule::AtLeastNOf(2, vec![l1.train.clone(), l2.train.clone()]),
                train_metric: Rule::AtLeastNOf(
                    2,
                    vec![l1.train_metric.clone(), l2.train_metric.clone()],
                ),
                test: Rule::AtLeastNOf(2, vec![l1.test.clone(), l2.test.clone()]),
                test_metric: Rule::AtLeastNOf(
                    2,
                    vec![l1.test_metric.clone(), l2.test_metric.clone()],
                ),
                list: Rule::AtLeastNOf(2, vec![l1.list.clone(), l2.list.clone()]),
                fetch: Rule::AtLeastNOf(2, vec![l1.fetch.clone(), l2.fetch.clone()]),
                delete: Rule::AtLeastNOf(2, vec![l1.delete.clone(), l2.delete.clone()]),
                result_strategy: l1.result_strategy.clone(),
            },
            ResultStrategy::Or => License {
                train: Rule::AtLeastNOf(1, vec![l1.train.clone(), l2.train.clone()]),
                train_metric: Rule::AtLeastNOf(
                    1,
                    vec![l1.train_metric.clone(), l2.train_metric.clone()],
                ),
                test: Rule::AtLeastNOf(1, vec![l1.test.clone(), l2.test.clone()]),
                test_metric: Rule::AtLeastNOf(
                    1,
                    vec![l1.test_metric.clone(), l2.test_metric.clone()],
                ),
                list: Rule::AtLeastNOf(1, vec![l1.list.clone(), l2.list.clone()]),
                fetch: Rule::AtLeastNOf(1, vec![l1.fetch.clone(), l2.fetch.clone()]),
                delete: Rule::AtLeastNOf(1, vec![l1.delete.clone(), l2.delete.clone()]),
                result_strategy: l1.result_strategy.clone(),
            },
            ResultStrategy::Custom(_) => {
                if l1 == l2 {
                    l1.clone()
                } else {
                    return Err(Status::permission_denied(
                        "Checkpoint and dataset licenses are not combatible",
                    ));
                }
            }
        })
    }
    pub fn train_metric(&self) -> Rule {
        self.train_metric.clone()
    }
    pub fn test_metric(&self) -> Rule {
        self.test_metric.clone()
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn t() {
        let license = License {
            train: Rule::SignedWith(vec![0, 1, 2]),
            result_strategy: ResultStrategy::And,
        };

        let a = serde_cbor::to_vec(license);
    }
}
