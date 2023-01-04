use crate::session::SessionManager;
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tonic::Status;

#[derive(Debug)]
pub struct Tracking {
    sess_manager: Arc<SessionManager>,
    storage_dos_tracking: Arc<RwLock<HashMap<String, Vec<Instant>>>>,
    memory_dos_tracking: Arc<RwLock<HashMap<String, Vec<Instant>>>>,
    max_saves: usize,
    max_runs: usize,
}

impl Tracking {
    pub fn new(sess_manager: Arc<SessionManager>, max_saves: usize, max_runs: usize) -> Self {
        Self {
            sess_manager,
            storage_dos_tracking: Arc::new(RwLock::new(HashMap::new())),
            memory_dos_tracking: Arc::new(RwLock::new(HashMap::new())),
            max_saves,
            max_runs,
        }
    }

    pub fn dos_check(
        &self,
        timer: u64,
        user_id: String,
        token: Option<Bytes>,
        operation: &str,
    ) -> Result<(), Status> {
        let mut dos_tracking = if operation == "run" {
            self.memory_dos_tracking.write().unwrap()
        } else {
            self.storage_dos_tracking.write().unwrap()
        };

        let max = if operation == "run" {
            self.max_runs
        } else {
            self.max_saves
        };

        let runs = dos_tracking.get(&user_id);
        let mut runs_vec: Vec<Instant>;
        match runs {
            Some(runs) => {
                if runs.len() > max {
                    if runs[runs.len() - max].elapsed().as_secs() < timer {
                        self.sess_manager.delete_session(token.unwrap());
                        self.sess_manager
                            .block_user(user_id.clone(), Instant::now());
                        dos_tracking.remove(&user_id);
                        return Err(Status::unknown(
                            "DoS attempt detected! Your access is temporarily blocked.",
                        ));
                    }
                }
                runs_vec = runs.to_vec();
                runs_vec.push(Instant::now());
            }
            None => {
                runs_vec = vec![Instant::now()];
            }
        }
        dos_tracking.insert(user_id.clone(), runs_vec);
        Ok(())
    }
}
