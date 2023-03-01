use crate::session::SessionManager;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use tonic::Status;

#[derive(Debug)]
pub struct Tracking {
    sess_manager: Arc<SessionManager>,
    //Maps users to their total consumption and a hashmap of their dfs and their sizes
    pub memory_quota: Arc<RwLock<HashMap<String, (usize, HashMap<String, usize>)>>>,
    max_memory: Mutex<usize>,
    pub dataframe_user: Arc<RwLock<HashMap<String, String>>>, //Maps dataframe identifiers to users
}

impl Tracking {
    pub fn new(sess_manager: Arc<SessionManager>, max_memory: usize) -> Self {
        Self {
            sess_manager,
            memory_quota: Arc::new(RwLock::new(HashMap::new())),
            max_memory: Mutex::new(max_memory),
            dataframe_user: Arc::new(RwLock::new(HashMap::new())),
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
                        "You have consumed your entire memory quota. Please delete some of your dataframes to free memory.",
                    ));
                }
                let mut identifiers = identifiers.to_owned();
                identifiers.insert(identifier.clone(), size);
                (consumption + size, identifiers)
            }
            None => {
                let mut hash_map = HashMap::new();
                hash_map.insert(identifier.clone(), size);
                (size, hash_map)
            }
        };
        memory_quota.insert(user_id.clone(), resulting_consumption);
        let mut dataframe_user = self.dataframe_user.write().unwrap();
        dataframe_user.insert(identifier, user_id);
        Ok(())
    }
}
