use bastionlab_common::session_proto::ClientInfo;
use bastionlab_learning::serialization::SizedObjectsBytes;
use ring::hmac;
use std::convert::TryInto;
use std::sync::{Arc, RwLock};
use tch::TchError;

/// Stored object with name, description and owner key
#[derive(Debug)]
pub struct Artifact<T> {
    pub data: Arc<RwLock<T>>,
    pub name: String,
    pub description: String,
    pub secret: hmac::Key,
    pub meta: Vec<u8>,
    pub client_info: Option<ClientInfo>,
}

impl<T: Default> Default for Artifact<T> {
    fn default() -> Self {
        Self {
            description: String::default(),
            name: String::default(),
            secret: hmac::Key::new(ring::hmac::HMAC_SHA256, &[0]),
            meta: Vec::default(),
            client_info: None,
            data: Arc::default(),
        }
    }
}

// impl<T> Artifact<T> {
//     /// Verifies passed meassage and tag against stored owner key.
//     pub fn verify(&self, msg: &[u8], tag: &[u8]) -> bool {
//         match hmac::verify(&self.secret, msg, tag) {
//             Ok(()) => true,
//             Err(_) => false,
//         }
//     }
// }

impl<T> Artifact<T>
where
    for<'a> &'a T: TryInto<SizedObjectsBytes, Error = TchError>,
{
    /// Serializes the contained object and returns a new artifact that contains
    /// a SizedObjectBytes (binary buffer) instead of the object.
    ///
    /// Note that the object should be convertible into a SizedObjectBytes (with `TryInto`).
    pub fn serialize(&self) -> Result<Artifact<SizedObjectsBytes>, TchError> {
        Ok(Artifact {
            data: Arc::new(RwLock::new((&*self.data.read().unwrap()).try_into()?)),
            name: self.name.clone(),
            description: self.description.clone(),
            secret: self.secret.clone(),
            meta: self.meta.clone(),
            client_info: self.client_info.clone(),
        })
    }
}

impl Artifact<SizedObjectsBytes> {
    /// Deserializes the contained [`SizedObjectBytes`] object (binary buffer) and returns a
    /// new artifact that contains the deserialized object instead.
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
            meta: self.meta,
            client_info: self.client_info,
        })
    }
}
