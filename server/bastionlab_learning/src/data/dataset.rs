use super::privacy_guard::{BatchDependence, PrivacyBudget, PrivacyContext, PrivacyGuard};
use crate::serialization::SizedObjectsBytes;
use rand::{seq::SliceRandom, thread_rng};
use std::convert::TryFrom;
use std::io::Cursor;
use std::ops::Deref;
use std::sync::{Arc, Mutex, RwLock};
use tch::{Device, IndexOp, TchError, Tensor};

/// Simple in-memory dataset that keeps track of its usage in terms of privacy budget
#[derive(Debug)]
pub struct Dataset {
    samples_inputs: Vec<Mutex<Tensor>>,
    labels: Mutex<Tensor>,
    privacy_context: Arc<RwLock<PrivacyContext>>,
}

/// Simple iterator over [`Dataset`].
pub struct DatasetIter<'a> {
    dataset: &'a Dataset,
    indexes: Vec<i64>,
    batch_size: usize,
    batch_id: usize,
}

impl<'a> Iterator for DatasetIter<'a> {
    type Item = (Vec<PrivacyGuard<Tensor>>, PrivacyGuard<Tensor>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.indexes.len() < self.batch_size {
            None
        } else {
            let indexes = self.indexes.drain(..self.batch_size);
            let samples_inputs_guards: Vec<_> = self
                .dataset
                .samples_inputs
                .iter()
                .map(|input| input.lock().unwrap())
                .collect();
            let labels_guard = self.dataset.labels.lock().unwrap();
            let items = indexes.map(|idx| {
                (
                    samples_inputs_guards
                        .iter()
                        .map(|input| input.i(idx))
                        .collect::<Vec<_>>(),
                    labels_guard.i(idx),
                )
            });
            let mut batch_inputs = Vec::with_capacity(samples_inputs_guards.len());
            for _ in 0..samples_inputs_guards.len() {
                batch_inputs.push(Vec::with_capacity(self.batch_size));
            }
            let mut batch_labels = Vec::with_capacity(self.batch_size);
            for (inputs, label) in items {
                for (batch_input, input) in batch_inputs.iter_mut().zip(inputs) {
                    batch_input.push(input);
                }
                batch_labels.push(label);
            }
            let batch_id = self.batch_id;
            self.batch_id += 1;
            let batch_inputs: Vec<_> = batch_inputs
                .iter()
                .map(|input| {
                    PrivacyGuard::new(
                        Tensor::f_stack(&input, 0).unwrap(), // Handle this in a better way
                        BatchDependence::Independent(vec![batch_id]),
                        Arc::clone(&self.dataset.privacy_context),
                    )
                })
                .collect();
            let batch_labels = PrivacyGuard::new(
                Tensor::f_stack(&batch_labels, 0).unwrap(), // Handle this in a better way
                BatchDependence::Independent(vec![batch_id]),
                Arc::clone(&self.dataset.privacy_context),
            );
            Some((batch_inputs, batch_labels))
        }
    }
}

impl Dataset {
    /// Returns an iterator over this dataset.
    pub fn iter_shuffle<'a>(&'a self, batch_size: usize) -> DatasetIter<'a> {
        let mut rng = thread_rng();
        let mut indexes: Vec<_> = (0..self.len() as i64).collect();
        indexes.shuffle(&mut rng);
        DatasetIter {
            dataset: self,
            indexes,
            batch_size,
            batch_id: 0,
        }
    }
    pub fn iter<'a>(&'a self, batch_size: usize) -> DatasetIter<'a> {
        let indexes: Vec<_> = (0..self.len() as i64).collect();
        DatasetIter {
            dataset: self,
            indexes,
            batch_size,
            batch_id: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.labels.lock().unwrap().size()[0] as usize
    }
}

impl TryFrom<SizedObjectsBytes> for Dataset {
    type Error = TchError;

    fn try_from(value: SizedObjectsBytes) -> Result<Self, Self::Error> {
        let mut samples_inputs: Vec<Option<Mutex<Tensor>>> = Vec::new();
        let mut labels: Option<Mutex<Tensor>> = None;
        #[allow(unused_variables)]
        let mut privacy_limit = PrivacyBudget::Private(0.0);

        for object in value {
            let data =
                Tensor::load_multi_from_stream_with_device(Cursor::new(object), Device::Cpu)?;
            for (name, tensor) in data {
                match &*name {
                    "labels" => match &labels {
                        Some(labels) => {
                            let mut labels = labels.lock().unwrap();
                            *labels = Tensor::f_cat(&[&*labels, &tensor], 0)?;
                        }
                        None => {
                            labels = Some(Mutex::new(tensor));
                        }
                    },
                    "privacy_limit" => {
                        let limit = tensor.f_double_value(&[])? as f32;
                        #[allow(unused_assignments)]
                        {
                            privacy_limit = if limit < 0.0 {
                                PrivacyBudget::NotPrivate
                            } else {
                                PrivacyBudget::Private(limit)
                            };
                        }
                    }
                    s => {
                        if s.starts_with("samples_") {
                            let idx: usize = s[8..].parse().or(Err(TchError::FileFormat(
                                String::from(format!("Invalid data, unknown field {}.", s)),
                            )))?;
                            if samples_inputs.len() <= idx {
                                for _ in samples_inputs.len()..(idx + 1) {
                                    samples_inputs.push(None);
                                }
                            }
                            match &samples_inputs[idx] {
                                Some(samples_input) => {
                                    let mut samples_input = samples_input.lock().unwrap();
                                    *samples_input = Tensor::f_cat(&[&*samples_input, &tensor], 0)?;
                                }
                                None => {
                                    samples_inputs[idx] = Some(Mutex::new(tensor));
                                }
                            }
                        } else {
                            return Err(TchError::FileFormat(String::from(format!(
                                "Invalid data, unknown field {}.",
                                s
                            ))));
                        }
                    }
                };
            }
        }
        let labels = labels.unwrap();
        let label_size = labels.lock().unwrap().size();
        let nb_samples = if label_size.len() > 0 {
            Ok(label_size[0])
        } else {
            Err(TchError::Kind(String::from(
                "Labels tensor has no dimmensions, cannot infer dataset size.",
            )))
        }?;
        Ok(Dataset {
            samples_inputs: samples_inputs.into_iter().map(|opt| opt.unwrap()).collect(),
            labels,
            privacy_context: Arc::new(RwLock::new(PrivacyContext::new(
                privacy_limit,
                nb_samples as usize,
            ))),
        })
    }
}

impl TryFrom<&Dataset> for SizedObjectsBytes {
    type Error = TchError;

    fn try_from(value: &Dataset) -> Result<Self, Self::Error> {
        let mut dataset_bytes = SizedObjectsBytes::new();
        let mut buf = Vec::new();
        let guards: Vec<_> = value
            .samples_inputs
            .iter()
            .map(|input| input.lock().unwrap())
            .collect();
        let mut named_tensors: Vec<_> = guards
            .iter()
            .enumerate()
            .map(|(i, input)| (format!("samples_{}", i), input.deref()))
            .collect();
        let labels = &*value.labels.lock().unwrap();
        named_tensors.push((String::from("labels"), labels));
        Tensor::save_multi_to_stream(&named_tensors, &mut buf)?;
        dataset_bytes.append_back(buf);

        Ok(dataset_bytes)
    }
}
