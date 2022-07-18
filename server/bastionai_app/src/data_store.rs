use anyhow::{anyhow, Result};
use ring::digest;
use std::collections::{hash_map::Entry, HashMap};
use std::sync::RwLock;
use tch::{Tensor, TrainableCModule};
use uuid::Uuid;
pub type ModuleType = HashMap<Uuid, Artifact<TrainableCModule>>;
pub type BatchType = HashMap<Uuid, Artifact<Batch>>;

#[derive(Debug)]
struct Permission {
    owner: bool, // The only permission
    user: bool,
    /* owner_id, replace with all elements */
}

#[derive(Debug)]
struct Meta {
    size: usize,
    chunk_size: usize,
    nb_chunks: usize,
}
pub struct Batch {
    tensors: Vec<Tensor>,
}

#[derive(Debug)]
pub struct Artifact<T> {
    permission: Permission,
    data: T,
}

unsafe impl<T> Sync for Artifact<T> {}
unsafe impl Sync for DataStore {}

struct InnerModules {
    module_by_id: ModuleType,
    module_hash_and_id: HashMap<Vec<u8>, Uuid>,
}

struct InnerBatch {
    batch_by_id: BatchType,
    batch_by_hash_and_id: HashMap<Vec<u8>, Uuid>,
}

pub struct DataStore {
    inner_modules: RwLock<InnerModules>,
    inner_batches: RwLock<InnerBatch>,
    /* Hashmap<user_id (username), pubKey> */
}

impl DataStore {
    pub fn new() -> DataStore {
        DataStore {
            inner_modules: RwLock::new(InnerModules {
                module_by_id: HashMap::new(),
                module_hash_and_id: HashMap::new(),
            }),
            inner_batches: RwLock::new(InnerBatch {
                batch_by_id: HashMap::new(),
                batch_by_hash_and_id: HashMap::new(),
            }),
        }
    }

    pub fn add_module_artifact(
        &self,
        artifacts: TrainableCModule,
        artifacts_bytes: &[u8],
    ) -> Option<Uuid> {
        let mut modules = self.inner_modules.write().unwrap();
        let id = Uuid::new_v4();
        let module_hash = digest::digest(&digest::SHA256, artifacts_bytes)
            .as_ref()
            .to_vec();

        {
            let module_id = match modules.module_hash_and_id.entry(module_hash) {
                Entry::Occupied(entry) => *entry.into_mut(),
                Entry::Vacant(entry) => {
                    entry.insert(id);
                    id
                }
            };

            let _ = match modules.module_by_id.entry(module_id) {
                Entry::Occupied(_) => return None,
                Entry::Vacant(entry) => entry.insert(Artifact {
                    permission: Permission {
                        owner: true,
                        user: true,
                    },
                    data: artifacts,
                }),
            };
        }

        Some(id)
    }

    pub fn add_batch_artifact(
        &self,
        artifacts: Vec<Tensor>,
        artifacts_bytes: &[u8],
    ) -> Option<String> {
        let mut batches = self.inner_batches.write().unwrap();
        let id = Uuid::new_v4();
        let batch_hash = digest::digest(&digest::SHA256, artifacts_bytes)
            .as_ref()
            .to_vec();

        {
            let batch_id = match batches.batch_by_hash_and_id.entry(batch_hash) {
                Entry::Occupied(entry) => *entry.into_mut(),
                Entry::Vacant(entry) => {
                    entry.insert(id);
                    id
                }
            };

            let _ = match batches.batch_by_id.entry(batch_id) {
                Entry::Occupied(_) => return None,
                Entry::Vacant(entry) => entry.insert(Artifact {
                    permission: Permission {
                        owner: true,
                        user: true,
                    },
                    data: Batch { tensors: artifacts },
                }),
            };
        }

        Some(id.to_string())
    }
}
