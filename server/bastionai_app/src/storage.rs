use ring::hmac;
use std::convert::TryInto;
use std::sync::{Arc, RwLock};
use bastionai_learning::serialization::SizedObjectsBytes;
use tch::TchError;

/// Stored object with encryption and owner key
#[derive(Debug)]
pub struct Artifact<T> {
    pub data: Arc<RwLock<T>>,
    pub name: String,
    pub description: String,
    pub secret: hmac::Key,
}

impl<T> Artifact<T> {
    /// Creates new artifact from data, description and owner key.
    pub fn new(data: T, name: String, description: String, secret: &[u8]) -> Self {
        Artifact {
            data: Arc::new(RwLock::new(data)),
            name,
            description,
            secret: hmac::Key::new(hmac::HMAC_SHA256, &secret),
        }
    }

    /// Verifies passed meaasge and tag against stored owner key.
    pub fn verify(&self, msg: &[u8], tag: &[u8]) -> bool {
        match hmac::verify(&self.secret, msg, tag) {
            Ok(()) => true,
            Err(_) => false,
        }
    }
}

impl<T> Artifact<T>
where
    for<'a> &'a T: TryInto<SizedObjectsBytes, Error = TchError>,
{
    /// Serializes the contained object and returns a new artifact that contains
    /// a SizedObjectBytes instead of the object.
    ///
    /// Note that the object should be convertible into a SizedObjectBytes (with `TryInto`).
    pub fn serialize(&self) -> Result<Artifact<SizedObjectsBytes>, TchError> {
        Ok(Artifact {
            data: Arc::new(RwLock::new((&*self.data.read().unwrap()).try_into()?)),
            name: self.name.clone(),
            description: self.description.clone(),
            secret: self.secret.clone(),
        })
    }
}

impl Artifact<SizedObjectsBytes> {
    /// Deserializes the contained [`SizedObjectBytes`] object and returns a new artifact that contains
    /// a the deserialized object instead.
    ///
    /// Note that the object should be convertible from a SizedObjectBytes (with `TryForm`).
    pub fn deserialize<T: TryFrom<SizedObjectsBytes, Error = TchError> + std::fmt::Debug>(
        self,
    ) -> Result<Artifact<T>, TchError> {
        Ok(Artifact {
            data: Arc::new(RwLock::new(T::try_from(
                Arc::try_unwrap(self.data).unwrap().into_inner().unwrap(),
            )?)),
            name: self.name,
            description: self.description,
            secret: self.secret,
        })
    }
}
