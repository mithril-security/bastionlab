use rand::{seq::SliceRandom, thread_rng};
use std::convert::TryFrom;
use std::io::Cursor;
use std::ops::Deref;
use std::sync::Mutex;
use std::sync::{Arc, RwLock};
use tch::{Device, IndexOp, TchError, Tensor};
use crate::serialization::SizedObjectsBytes;

/// Simple in-memory dataset
#[derive(Debug)]
pub struct Dataset {
    samples_inputs: Vec<Mutex<Tensor>>,
    labels: Mutex<Tensor>,
}

/// Simple iterator over [`Dataset`].
pub struct DatasetIter {
    dataset: Arc<RwLock<Dataset>>,
    indexes: Vec<i64>,
    batch_size: usize,
}

impl Iterator for DatasetIter {
    type Item = (Vec<Tensor>, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.indexes.len() < self.batch_size {
            None
        } else {
            let indexes = self.indexes.drain(..self.batch_size);
            let dataset = self.dataset.read().unwrap();
            let samples_inputs_guards: Vec<_> = dataset.samples_inputs.iter().map(|input| input.lock().unwrap()).collect();
            let labels_guard = dataset.labels.lock().unwrap();
            let items = indexes.map(|idx| (samples_inputs_guards.iter().map(|input| input.i(idx)).collect::<Vec<_>>(), labels_guard.i(idx)));
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
            let batch_inputs: Vec<_> = batch_inputs.iter().map(|input| Tensor::stack(&input, 0)).collect();
            let batch_labels = Tensor::stack(&batch_labels, 0);
            Some((batch_inputs, batch_labels))
        }
    }
}

impl Dataset {
    /// Returns an iterator over this dataset.
    pub fn iter_shuffle(s: Arc<RwLock<Self>>, batch_size: usize) -> DatasetIter {
        let mut rng = thread_rng();
        let mut indexes: Vec<_> = (0..s.read().unwrap().len() as i64).collect();
        indexes.shuffle(&mut rng);
        DatasetIter {
            dataset: s,
            indexes,
            batch_size,
        }
    }
    pub fn iter(s: Arc<RwLock<Self>>, batch_size: usize) -> DatasetIter {
        let indexes: Vec<_> = (0..s.read().unwrap().len() as i64).collect();
        DatasetIter {
            dataset: s,
            indexes,
            batch_size,
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
        
        for object in value {
            let data =
                Tensor::load_multi_from_stream_with_device(Cursor::new(object), Device::Cpu)?;
            for (name, tensor) in data {
                match &*name {
                    "labels" => {
                        match &labels {
                            Some(labels) => {
                                let mut labels = labels.lock().unwrap();
                                *labels = Tensor::f_cat(&[&*labels, &tensor], 0)?;
                            }
                            None => {
                                labels = Some(Mutex::new(tensor));
                            }
                        }
                    }
                    s => if s.starts_with("samples_") {
                        let idx: usize = s[8..].parse()
                            .or(Err(TchError::FileFormat(String::from(format!("Invalid data, unknown field {}.", s)))))?;
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
                        return Err(TchError::FileFormat(String::from(format!("Invalid data, unknown field {}.", s))))
                    }
                };
            }
        }
        Ok(Dataset {
            samples_inputs: samples_inputs.into_iter().map(|opt| opt.unwrap()).collect(),
            labels: labels.unwrap(),
        })
    }
}

impl TryFrom<&Dataset> for SizedObjectsBytes {
    type Error = TchError;

    fn try_from(value: &Dataset) -> Result<Self, Self::Error> {
        let mut dataset_bytes = SizedObjectsBytes::new();
        let mut buf = Vec::new();
        let guards: Vec<_> = value.samples_inputs.iter().map(|input| input.lock().unwrap()).collect();
        let mut named_tensors: Vec<_> = guards.iter().enumerate().map(|(i, input)| (format!("samples_{}", i), input.deref())).collect();
        let labels = &*value.labels.lock().unwrap();
        named_tensors.push((String::from("labels"), labels));
        Tensor::save_multi_to_stream(&named_tensors, &mut buf)?;
        dataset_bytes.append_back(buf);

        Ok(dataset_bytes)
    }
}
