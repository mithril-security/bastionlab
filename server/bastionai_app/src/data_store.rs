use anyhow::{anyhow, Result};
use ring::digest;
use std::collections::{hash_map::Entry, HashMap};
use std::sync::RwLock;
use tch::{Tensor, TrainableCModule};
use uuid::Uuid;
pub type ModuleType = HashMap<String, Artifact<TrainableCModule>>;
pub type BatchType = HashMap<String, Artifact<Batch>>;

#[derive(Debug)]
struct Permission {
    owner: bool,
    user: bool,
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

struct Group {
    read: bool,
    write: bool,
    execute: bool,
}

struct InnerModules {
    module_by_id: ModuleType,
    module_by_hash: HashMap<Vec<u8>, TrainableCModule>,
}

struct InnerBatch {
    batch_by_id: BatchType,
    batch_by_hash: HashMap<Vec<u8>, Batch>,
}

pub struct DataStore {
    inner_modules: RwLock<InnerModules>,
    inner_batches: RwLock<InnerBatch>,
    groups: Vec<Group>,
}

impl DataStore {
    pub fn new() -> DataStore {
        DataStore {
            inner_modules: RwLock::new(InnerModules {
                module_by_id: HashMap::new(),
                module_by_hash: HashMap::new(),
            }),
            inner_batches: RwLock::new(InnerBatch {
                batch_by_id: HashMap::new(),
                batch_by_hash: HashMap::new(),
            }),
            groups: Vec::new(),
        }
    }

    pub fn add_module_artifact(
        &self,
        id: Uuid,
        artifacts: TrainableCModule,
        artifacts_bytes: &[u8],
    ) -> Result<String> {
        let mut modules = self.inner_modules.write().unwrap();
        let module_hash = digest::digest(&digest::SHA256, artifacts_bytes)
            .as_ref()
            .to_vec();

        let module = match modules.module_by_hash.entry(module_hash) {
            Entry::Occupied(_) => Artifact {
                permission: Permission {
                    owner: true,
                    user: true,
                },
                data: artifacts,
            },

            Entry::Vacant(_) => Artifact {
                permission: Permission {
                    owner: true,
                    user: true,
                },
                data: artifacts,
            },
        };
        modules.module_by_id.insert(id.to_string(), module);

        Ok(id.to_string())
    }

    pub fn add_batch_artifact(
        &self,
        id: Uuid,
        artifacts: Vec<Tensor>,
        artifacts_bytes: &[u8],
    ) -> Result<String> {
        let mut batches = self.inner_batches.write().unwrap();

        let batch_hash = digest::digest(&digest::SHA256, artifacts_bytes)
            .as_ref()
            .to_vec();

        let batch = match batches.batch_by_hash.entry(batch_hash) {
            Entry::Occupied(_) => Batch { tensors: artifacts },
            Entry::Vacant(_) => Batch { tensors: artifacts },
        };

        batches.batch_by_id.insert(
            id.to_string(),
            Artifact {
                permission: Permission {
                    owner: true,
                    user: true,
                },
                data: batch,
            },
        );
        Ok(id.to_string())
    }
}
