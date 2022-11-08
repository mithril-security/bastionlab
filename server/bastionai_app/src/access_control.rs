use crate::remote_torch;
use anyhow::Result;
use prost::Message;
use ring::{digest, signature};
use tonic::{metadata::MetadataMap, Request, Status};

pub use crate::remote_torch::{
    result_strategy::Strategy as RStrategyKind, rule::AtLeastNOf, rule::Rule as IRule, License,
    ResultStrategy, Rule,
};

impl Rule {
    fn verify(
        &self,
        header: &MetadataMap,
        message: &[u8],
        user_public_key_hash: Option<&[u8]>,
        checkpoint_hash: Option<&[u8]>,
        dataset_hash: Option<&[u8]>,
    ) -> Result<(), Status> {
        match self
            .rule
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("malformated rule"))?
        {
            IRule::AtLeastNOf(AtLeastNOf { n, rules }) => {
                let mut m = 0;
                let mut failed = String::new();
                for (i, policy) in rules.iter().enumerate() {
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
            IRule::SignedWith(raw_public_key) => {
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
            IRule::WithCheckpoint(hash) => {
                if let Some(checkpoint_hash) = checkpoint_hash {
                    if &*hash != checkpoint_hash {
                        return Err(Status::permission_denied("Checkpoint mismatch"));
                    }
                }
                Ok(())
            }
            IRule::WithDataset(hash) => {
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

impl License {
    pub fn verify_fetch(
        &self,
        req: &Request<remote_torch::ReferenceRequest>,
    ) -> Result<(), Status> {
        self.fetch
            .as_ref()
            .ok_or_else(|| Status::permission_denied("license does not permit fetch"))?
            .verify(
                &req.metadata(),
                &get_message(b"fetch", req)?,
                None,
                None,
                None,
            )
    }
    pub fn verify_list(&self, req: &Request<remote_torch::Empty>) -> Result<(), Status> {
        self.list
            .as_ref()
            .ok_or_else(|| Status::permission_denied("license does not permit list"))?
            .verify(
                &req.metadata(),
                &get_message(b"list", req)?,
                None,
                None,
                None,
            )
    }
    pub fn verify_delete(
        &self,
        req: &Request<remote_torch::ReferenceRequest>,
    ) -> Result<(), Status> {
        self.delete
            .as_ref()
            .ok_or_else(|| Status::permission_denied("license does not permit delete"))?
            .verify(
                &req.metadata(),
                &get_message(b"delete", req)?,
                None,
                None,
                None,
            )
    }
    pub fn verify_train(&self, req: &Request<remote_torch::TrainRequest>) -> Result<(), Status> {
        self.train
            .as_ref()
            .ok_or_else(|| Status::permission_denied("license does not permit train"))?
            .verify(
                &req.metadata(),
                &get_message(b"train", req)?,
                None,
                Some(&req.get_ref().model[..]),
                Some(&req.get_ref().dataset[..]),
            )
    }
    pub fn verify_test(&self, req: &Request<remote_torch::TestRequest>) -> Result<(), Status> {
        self.test
            .as_ref()
            .ok_or_else(|| Status::permission_denied("license does not permit test"))?
            .verify(
                &req.metadata(),
                &get_message(b"test", req)?,
                None,
                Some(&req.get_ref().model[..]),
                Some(&req.get_ref().dataset[..]),
            )
    }
    pub fn combine(&self, other: &License) -> Result<License, Status> {
        let l1 = if self
            .result_strategy
            .as_ref()
            .ok_or_else(|| Status::permission_denied("no result strategy"))?
            .strategy()
            == RStrategyKind::Custom
        {
            self.result_strategy
                .as_ref()
                .unwrap()
                .custom_license
                .as_ref()
                .ok_or_else(|| Status::permission_denied("no custom license in result strategy"))?
        } else {
            self
        };
        let l2 = if other
            .result_strategy
            .as_ref()
            .ok_or_else(|| Status::permission_denied("no result strategy"))?
            .strategy()
            == RStrategyKind::Custom
        {
            other
                .result_strategy
                .as_ref()
                .unwrap()
                .custom_license
                .as_ref()
                .ok_or_else(|| Status::permission_denied("no custom license in result strategy"))?
        } else {
            other
        };

        if l1.result_strategy != l2.result_strategy {
            return Err(Status::permission_denied(
                "Checkpoint and dataset licenses are not combatible",
            ));
        }

        let make_rule = |n: u64, r1: Option<Rule>, r2: Option<Rule>| -> Result<_, Status> {
            Ok(Some(Rule {
                rule: Some(IRule::AtLeastNOf(AtLeastNOf {
                    n,
                    rules: vec![
                        r1.ok_or_else(|| Status::permission_denied("malformated rule"))?,
                        r2.ok_or_else(|| Status::permission_denied("malformated rule"))?,
                    ],
                })),
            }))
        };

        Ok(match l1.result_strategy.as_ref().unwrap().strategy() {
            RStrategyKind::Checkpoint => l1.clone(),
            RStrategyKind::Dataset => l2.clone(),
            RStrategyKind::And => License {
                train: make_rule(2, l1.train.clone(), l2.train.clone())?,
                test: make_rule(2, l1.test.clone(), l2.test.clone())?,
                list: make_rule(2, l1.list.clone(), l2.list.clone())?,
                fetch: make_rule(2, l1.fetch.clone(), l2.fetch.clone())?,
                delete: make_rule(2, l1.delete.clone(), l2.delete.clone())?,
                result_strategy: l1.result_strategy.clone(),
            },
            RStrategyKind::Or => License {
                train: make_rule(1, l1.train.clone(), l2.train.clone())?,
                test: make_rule(1, l1.test.clone(), l2.test.clone())?,
                list: make_rule(1, l1.list.clone(), l2.list.clone())?,
                fetch: make_rule(1, l1.fetch.clone(), l2.fetch.clone())?,
                delete: make_rule(1, l1.delete.clone(), l2.delete.clone())?,
                result_strategy: l1.result_strategy.clone(),
            },
            RStrategyKind::Custom => {
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
}
