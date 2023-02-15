use crate::session::SessionManager;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use tonic::Status;

#[derive(Debug)]
pub struct Tracking {
    sess_manager: Arc<SessionManager>,
    pub memory_quota: Arc<RwLock<HashMap<String, (usize, HashMap<String, usize>)>>>,
    max_memory: Mutex<usize>,
}

impl Tracking {
    pub fn new(sess_manager: Arc<SessionManager>, max_memory: usize) -> Self {
        Self {
            sess_manager,
            memory_quota: Arc::new(RwLock::new(HashMap::new())),
            max_memory: Mutex::new(max_memory),
        }
    }

    pub fn memory_quota_check(
        &self,
        size: usize,
        user_id: String,
        identifier: String,
    ) -> Result<(), Status> {
        //We return immediately if auth is disabled
        if !self.sess_manager.auth_enabled() {
            return Ok(());
        }

        let mut memory_quota = self.memory_quota.write().unwrap();
        let consumption = memory_quota.get(&user_id);
        let resulting_consumption = match consumption {
            Some((consumption, identifiers)) => {
                if consumption + size > *self.max_memory.lock().unwrap() {
                    return Err(Status::unknown(
                        "You have consumed your entire memory quota. Please ask the data owner to delete your dataframes to free memory.",
                    ));
                }
                let mut identifiers = identifiers.to_owned();
                identifiers.insert(identifier, size);
                (consumption + size, identifiers)
            }
            None => {
                let mut hash_map = HashMap::new();
                hash_map.insert(identifier, size);
                (size, hash_map)
            }
        };
        memory_quota.insert(user_id, resulting_consumption);
        Ok(())
    }
}
