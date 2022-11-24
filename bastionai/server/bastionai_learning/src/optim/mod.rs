use std::collections::HashMap;
use std::io::Cursor;

use tch::TchError;
use tch::Tensor;

mod adam;
mod optimizer;
mod sgd;

fn initialize_statistics() -> HashMap<String, Option<Tensor>> {
    let v = HashMap::new();
    v
}

/// Converts a [`HashMap<String, Option<Tensor>>`] to [`Result<Vec<u8>, TchError>`].
/// This uses [`Tensor::save_multi_to_stream`] to save the values in the statistics to
/// make it easy to load and store the state of the Optimizer.
fn stats_to_bytes<'a>(
    named_stats: &'a HashMap<String, Option<Tensor>>,
) -> Result<Vec<u8>, TchError> {
    let mut stats = HashMap::new();
    for (n, v) in named_stats {
        match v {
            Some(v) => stats.insert(n, v),
            None => None,
        };
    }

    let stats = stats.iter().map(|(&k, &v)| (k, v)).collect::<Vec<_>>();
    let mut buf: Vec<u8> = Vec::new();
    Tensor::save_multi_to_stream(stats.as_slice(), &mut buf)?;
    Ok(buf)
}

fn bytes_to_stats(bytes: &[u8]) -> Result<HashMap<String, Option<Tensor>>, TchError> {
    let mut stats = HashMap::new();
    let output = Tensor::load_multi_from_stream(Cursor::new(bytes))?;
    output.into_iter().for_each(|(k, v)| {
        stats.insert(k.clone(), Some(v));
    });
    Ok(stats)
}

pub use adam::Adam;
pub use optimizer::Optimizer;
pub use optimizer::OptimizerStateType;
pub use sgd::SGD;
