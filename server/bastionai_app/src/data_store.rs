use ring::digest;

use std::collections::{hash_map::Entry, HashMap};
use std::sync::RwLock;
use tch::{Tensor, TrainableCModule};
use uuid::Uuid;
pub type ModuleType = HashMap<Uuid, Artifact<TrainableCModule>>;
pub type BatchType = HashMap<Uuid, Artifact<Batch>>;

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
struct Permission {
    owner: bool, // The only permission
    user: bool,
}

#[derive(Debug, Clone)]
struct Meta {
    pub description: String,
}
#[allow(dead_code)]
pub struct Batch {
    tensors: Vec<Tensor>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct Artifact<T> {
    permission: Permission,
    data: T,
    meta: Meta,
}

impl<T> Artifact<T> {
    pub fn get_data(&self) -> &T {
        &self.data
    }
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
        description: &str,
    ) -> Option<Uuid> {
        let mut modules = self.inner_modules.write().unwrap();
        let module_hash = digest::digest(&digest::SHA256, artifacts_bytes)
            .as_ref()
            .to_vec();

        let module_id = match modules.module_hash_and_id.entry(module_hash) {
            Entry::Occupied(entry) => *entry.into_mut(),
            Entry::Vacant(entry) => {
                let id = Uuid::new_v4();
                entry.insert(id);
                id
            }
        };

        match modules.module_by_id.entry(module_id) {
            Entry::Occupied(_) => return None,
            Entry::Vacant(entry) => entry.insert(Artifact {
                permission: Permission {
                    owner: true,
                    user: true,
                },
                data: artifacts,
                meta: Meta {
                    description: description.to_string(),
                },
            }),
        };

        Some(module_id)
    }

    pub fn add_batch_artifact(
        &self,
        artifacts: Vec<Tensor>,
        artifacts_bytes: &[u8],
        description: &str,
    ) -> Option<Uuid> {
        let mut batches = self.inner_batches.write().unwrap();
        let batch_hash = digest::digest(&digest::SHA256, artifacts_bytes)
            .as_ref()
            .to_vec();

        let batch_id = match batches.batch_by_hash_and_id.entry(batch_hash) {
            Entry::Occupied(entry) => *entry.into_mut(),
            Entry::Vacant(entry) => {
                let id = Uuid::new_v4();
                entry.insert(id);
                id
            }
        };

        match batches.batch_by_id.entry(batch_id) {
            Entry::Occupied(_) => return None,
            Entry::Vacant(entry) => entry.insert(Artifact {
                permission: Permission {
                    owner: true,
                    user: true,
                },
                data: Batch { tensors: artifacts },
                meta: Meta {
                    description: description.to_string(),
                },
            }),
        };

        Some(batch_id)
    }

    pub fn delete_batch(&self, identifier: Uuid) -> Option<bool> {
        let mut batches = self.inner_batches.write().unwrap();
        let res = match batches.batch_by_id.remove(&identifier) {
            Some(_) => Some(identifier),
            None => None,
        };

        match res {
            Some(v) => {
                batches.batch_by_hash_and_id.retain(|_, x| *x != v);
                Some(true)
            }
            None => None,
        }
    }

    pub fn delete_model(&self, identifier: Uuid) -> Option<bool> {
        let mut modules = self.inner_modules.write().unwrap();
        let res = match modules.module_by_id.remove(&identifier) {
            Some(_) => Some(identifier),
            None => None,
        };

        match res {
            Some(v) => {
                modules.module_hash_and_id.retain(|_, x| *x != v);
                Some(true)
            }
            None => None,
        }
    }

    pub fn get_model_with_identifier<U>(
        &self,
        identifier: Uuid,
        func: impl Fn(&Artifact<TrainableCModule>) -> U,
    ) -> Option<U> {
        let modules = self.inner_modules.read().unwrap();
        match modules.module_by_id.get(&identifier) {
            Some(v) => Some(func(v)),
            None => None,
        }
    }

    fn get_available_objects<T>(
        &self,
        source: &HashMap<Uuid, Artifact<T>>,
    ) -> Vec<(String, String)> {
        let res = source
            .iter()
            .map(|(k, v)| (k.to_string(), v.meta.description.clone()))
            .collect::<Vec<(String, String)>>();
        res
    }

    pub fn get_available_models(&self) -> Vec<(String, String)> {
        let modules = self.inner_modules.read().unwrap();
        self.get_available_objects(&modules.module_by_id)
    }

    pub fn get_available_datasets(&self) -> Vec<(String, String)> {
        let batches = self.inner_batches.read().unwrap();
        self.get_available_objects(&batches.batch_by_id)
    }
}
