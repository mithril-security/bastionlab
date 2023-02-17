use ndarray::{Array, ArrayView, Axis, Dim, IxDynImpl};
use polars::prelude::DataFrame;
use tonic::Status;

use crate::common_conversions::{ndarray_to_df, to_status_error};

// FIXME: Try to update several impls with macros or generics to simplify implementation

#[derive(Debug, Clone)]
pub enum ArrayStore {
    AxdynI64(Array<i64, Dim<IxDynImpl>>),
    AxdynU64(Array<u64, Dim<IxDynImpl>>),
    AxdynU32(Array<u32, Dim<IxDynImpl>>),
    AxdynF64(Array<f64, Dim<IxDynImpl>>),
    AxdynF32(Array<f32, Dim<IxDynImpl>>),
    AxdynI32(Array<i32, Dim<IxDynImpl>>),
    AxdynI16(Array<i16, Dim<IxDynImpl>>),

    // Usize addition to support linfa easily without
    // casting to `usize`
    AxdynUsize(Array<usize, Dim<IxDynImpl>>),
}

/// This is used to shuffle the inner array by using the [`select`] function on ArrayBase.
///
/// It shuffling along the row axis.
fn shuffle<A>(array: &Array<A, Dim<IxDynImpl>>, indices: &[usize]) -> Array<A, Dim<IxDynImpl>>
where
    A: Copy + Clone,
{
    array.select(Axis(0), indices)
}

fn split<A>(
    array: &Array<A, Dim<IxDynImpl>>,
    ratios: (f64, f64),
) -> (Array<A, Dim<IxDynImpl>>, Array<A, Dim<IxDynImpl>>)
where
    A: Clone,
{
    let height = array.dim()[0];
    let upper = (height as f64 * ratios.0).floor() as usize;
    let lower = height - upper;

    let upper = array.select(Axis(0), &(0..upper).collect::<Vec<_>>()[..]);
    let lower = array.select(Axis(0), &((height - lower)..height).collect::<Vec<_>>()[..]);

    (upper, lower)
}

fn stack<A>(
    axis: Axis,
    arrays: &[Result<ArrayView<A, Dim<IxDynImpl>>, Status>],
) -> Result<Array<A, Dim<IxDynImpl>>, Status>
where
    A: Clone,
{
    let mut out_arrays = vec![];
    for array in arrays.iter() {
        let view = match array {
            Ok(v) => v,
            Err(e) => {
                return Err(Status::internal(e.to_string()));
            }
        };
        out_arrays.push(view.clone());
    }
    to_status_error(ndarray::stack::<A, Dim<IxDynImpl>>(axis, &out_arrays[..]))
}
impl ArrayStore {
    pub fn height(&self) -> usize {
        match self {
            ArrayStore::AxdynF32(a) => a.dim()[0],
            ArrayStore::AxdynI64(a) => a.dim()[0],
            ArrayStore::AxdynF64(a) => a.dim()[0],
            ArrayStore::AxdynI32(a) => a.dim()[0],
            ArrayStore::AxdynI16(a) => a.dim()[0],
            ArrayStore::AxdynUsize(a) => a.dim()[0],
            ArrayStore::AxdynU64(a) => a.dim()[0],
            ArrayStore::AxdynU32(a) => a.dim()[0],
        }
    }

    pub fn width(&self) -> usize {
        let columns = |dim: &[usize]| -> usize {
            if dim.len() == 1 {
                1
            } else {
                dim[1]
            }
        };
        match self {
            ArrayStore::AxdynF32(a) => columns(a.shape()),
            ArrayStore::AxdynI64(a) => columns(a.shape()),
            ArrayStore::AxdynF64(a) => columns(a.shape()),
            ArrayStore::AxdynI32(a) => columns(a.shape()),
            ArrayStore::AxdynI16(a) => columns(a.shape()),
            ArrayStore::AxdynUsize(a) => columns(a.shape()),
            ArrayStore::AxdynU64(a) => columns(a.shape()),
            ArrayStore::AxdynU32(a) => columns(a.shape()),
        }
    }

    pub fn shuffle(&self, indices: &[usize]) -> Self {
        match self {
            ArrayStore::AxdynF32(a) => Self::AxdynF32(shuffle::<f32>(a, indices)),
            ArrayStore::AxdynF64(a) => Self::AxdynF64(shuffle::<f64>(a, indices)),
            ArrayStore::AxdynI32(a) => Self::AxdynI32(shuffle::<i32>(a, indices)),
            ArrayStore::AxdynI64(a) => Self::AxdynI64(shuffle::<i64>(a, indices)),
            ArrayStore::AxdynI16(a) => Self::AxdynI16(shuffle::<i16>(a, indices)),
            ArrayStore::AxdynUsize(a) => Self::AxdynUsize(shuffle::<usize>(a, indices)),
            ArrayStore::AxdynU64(a) => Self::AxdynU64(shuffle::<u64>(a, indices)),
            ArrayStore::AxdynU32(a) => Self::AxdynU32(shuffle::<u32>(a, indices)),
        }
    }

    pub fn split(&self, ratios: (f64, f64)) -> (Self, Self) {
        /*
            Arrays could be split on a several axes but in this implementation, we are
            only splitting on the Oth Axis (row-wise).
        */
        match self {
            ArrayStore::AxdynI64(a) => {
                let (upper, lower) = split::<i64>(a, ratios);
                (ArrayStore::AxdynI64(upper), ArrayStore::AxdynI64(lower))
            }
            ArrayStore::AxdynF64(a) => {
                let (upper, lower) = split::<f64>(a, ratios);

                (ArrayStore::AxdynF64(upper), ArrayStore::AxdynF64(lower))
            }
            ArrayStore::AxdynF32(a) => {
                let (upper, lower) = split::<f32>(a, ratios);

                (ArrayStore::AxdynF32(upper), ArrayStore::AxdynF32(lower))
            }
            ArrayStore::AxdynI32(a) => {
                let (upper, lower) = split::<i32>(a, ratios);

                (ArrayStore::AxdynI32(upper), ArrayStore::AxdynI32(lower))
            }

            ArrayStore::AxdynI16(a) => {
                let (upper, lower) = split::<i16>(a, ratios);

                (ArrayStore::AxdynI16(upper), ArrayStore::AxdynI16(lower))
            }

            ArrayStore::AxdynUsize(a) => {
                let (upper, lower) = split::<usize>(a, ratios);

                (ArrayStore::AxdynUsize(upper), ArrayStore::AxdynUsize(lower))
            }

            ArrayStore::AxdynU64(a) => {
                let (upper, lower) = split::<u64>(a, ratios);

                (ArrayStore::AxdynU64(upper), ArrayStore::AxdynU64(lower))
            }
            ArrayStore::AxdynU32(a) => {
                let (upper, lower) = split::<u32>(a, ratios);

                (ArrayStore::AxdynU32(upper), ArrayStore::AxdynU32(lower))
            }
        }
    }

    pub fn append(&mut self, other: &Self, axis: Axis) -> Result<ArrayStore, Status> {
        let cannot_append =
            |a, b| return Err(Status::aborted(format!("Cannot append {a:?} to {b:?}")));
        let res = match self {
            ArrayStore::AxdynI64(a) => match other {
                Self::AxdynI64(b) => {
                    to_status_error(a.append(axis, b.view()))?;
                    Self::AxdynI64(a.clone())
                }
                _ => {
                    return cannot_append(self, other);
                }
            },
            ArrayStore::AxdynF64(a) => match other {
                Self::AxdynF64(b) => {
                    to_status_error(a.append(axis, b.view()))?;
                    Self::AxdynF64(a.clone())
                }
                _ => {
                    return cannot_append(self, other);
                }
            },
            ArrayStore::AxdynF32(a) => match other {
                Self::AxdynF32(b) => {
                    to_status_error(a.append(axis, b.view()))?;
                    Self::AxdynF32(a.clone())
                }
                _ => {
                    return cannot_append(self, other);
                }
            },
            ArrayStore::AxdynI32(a) => match other {
                Self::AxdynI32(b) => {
                    to_status_error(a.append(axis, b.view()))?;
                    Self::AxdynI32(a.clone())
                }
                _ => {
                    return cannot_append(self, other);
                }
            },
            ArrayStore::AxdynI16(a) => match other {
                Self::AxdynI16(b) => {
                    to_status_error(a.append(axis, b.view()))?;
                    Self::AxdynI16(a.clone())
                }
                _ => {
                    return cannot_append(self, other);
                }
            },
            ArrayStore::AxdynUsize(a) => match other {
                Self::AxdynUsize(b) => {
                    to_status_error(a.append(axis, b.view()))?;
                    Self::AxdynUsize(a.clone())
                }
                _ => {
                    return cannot_append(self, other);
                }
            },
            ArrayStore::AxdynU64(a) => match other {
                Self::AxdynU64(b) => {
                    to_status_error(a.append(axis, b.view()))?;
                    Self::AxdynU64(a.clone())
                }
                _ => {
                    return cannot_append(self, other);
                }
            },
            ArrayStore::AxdynU32(a) => match other {
                Self::AxdynU32(b) => {
                    to_status_error(a.append(axis, b.view()))?;
                    Self::AxdynU32(a.clone())
                }
                _ => {
                    return cannot_append(self, other);
                }
            },
        };

        Ok(res)
    }

    pub fn stack(axis: Axis, arrays: &[ArrayStore]) -> Result<ArrayStore, Status> {
        let first = arrays
            .get(0)
            .ok_or(Status::failed_precondition("Could not stack empty array"))?;

        let res = match first {
            ArrayStore::AxdynI64(_) => {
                let res = stack::<i64>(
                    axis,
                    &arrays
                        .iter()
                        .map(|v| match v {
                            ArrayStore::AxdynI64(a) => Ok(a.view()),
                            _ => {
                                return Err(Status::aborted(
                                    "DataTypes for all columns should be the same",
                                ));
                            }
                        })
                        .collect::<Vec<_>>()[..],
                );
                ArrayStore::AxdynI64(res?)
            }
            ArrayStore::AxdynI32(_) => {
                let res = stack::<i32>(
                    axis,
                    &arrays
                        .iter()
                        .map(|v| match v {
                            ArrayStore::AxdynI32(a) => Ok(a.view()),
                            _ => {
                                return Err(Status::aborted(
                                    "DataTypes for all columns should be the same",
                                ));
                            }
                        })
                        .collect::<Vec<_>>()[..],
                );
                ArrayStore::AxdynI32(res?)
            }
            ArrayStore::AxdynF64(_) => {
                let res = stack::<f64>(
                    axis,
                    &arrays
                        .iter()
                        .map(|v| match v {
                            ArrayStore::AxdynF64(a) => Ok(a.view()),
                            _ => {
                                return Err(Status::aborted(
                                    "DataTypes for all columns should be the same",
                                ));
                            }
                        })
                        .collect::<Vec<_>>()[..],
                );
                ArrayStore::AxdynF64(res?)
            }
            ArrayStore::AxdynF32(_) => {
                let res = stack::<f32>(
                    axis,
                    &arrays
                        .iter()
                        .map(|v| match v {
                            ArrayStore::AxdynF32(a) => Ok(a.view()),
                            _ => {
                                return Err(Status::aborted(
                                    "DataTypes for all columns should be the same",
                                ));
                            }
                        })
                        .collect::<Vec<_>>()[..],
                );
                ArrayStore::AxdynF32(res?)
            }
            ArrayStore::AxdynI16(_) => {
                let res = stack::<i16>(
                    axis,
                    &arrays
                        .iter()
                        .map(|v| match v {
                            ArrayStore::AxdynI16(a) => Ok(a.view()),
                            _ => {
                                return Err(Status::aborted(
                                    "DataTypes for all columns should be the same",
                                ));
                            }
                        })
                        .collect::<Vec<_>>()[..],
                );
                ArrayStore::AxdynI16(res?)
            }

            ArrayStore::AxdynUsize(_) => {
                let res = stack::<usize>(
                    axis,
                    &arrays
                        .iter()
                        .map(|v| match v {
                            ArrayStore::AxdynUsize(a) => Ok(a.view()),
                            _ => {
                                return Err(Status::aborted(
                                    "DataTypes for all columns should be the same",
                                ));
                            }
                        })
                        .collect::<Vec<_>>()[..],
                );
                ArrayStore::AxdynUsize(res?)
            }
            ArrayStore::AxdynU64(_) => {
                let res = stack::<u64>(
                    axis,
                    &arrays
                        .iter()
                        .map(|v| match v {
                            ArrayStore::AxdynU64(a) => Ok(a.view()),
                            _ => {
                                return Err(Status::aborted(
                                    "DataTypes for all columns should be the same",
                                ));
                            }
                        })
                        .collect::<Vec<_>>()[..],
                );
                ArrayStore::AxdynU64(res?)
            }
            ArrayStore::AxdynU32(_) => {
                let res = stack::<u32>(
                    axis,
                    &arrays
                        .iter()
                        .map(|v| match v {
                            ArrayStore::AxdynU32(a) => Ok(a.view()),
                            _ => {
                                return Err(Status::aborted(
                                    "DataTypes for all columns should be the same",
                                ));
                            }
                        })
                        .collect::<Vec<_>>()[..],
                );
                ArrayStore::AxdynU32(res?)
            }
        };

        Ok(res)
    }

    pub fn to_dataframe(&self, col_names: Vec<&str>) -> Result<DataFrame, Status> {
        match self {
            ArrayStore::AxdynI64(a) => ndarray_to_df::<i64, Dim<IxDynImpl>>(a, col_names),
            ArrayStore::AxdynF64(a) => ndarray_to_df::<f64, Dim<IxDynImpl>>(a, col_names),
            ArrayStore::AxdynF32(a) => ndarray_to_df::<f32, Dim<IxDynImpl>>(a, col_names),
            ArrayStore::AxdynI32(a) => ndarray_to_df::<i32, Dim<IxDynImpl>>(a, col_names),
            ArrayStore::AxdynU32(a) => ndarray_to_df::<u32, Dim<IxDynImpl>>(a, col_names),
            ArrayStore::AxdynU64(a) => ndarray_to_df::<u64, Dim<IxDynImpl>>(a, col_names),
            _ => {
                return Err(Status::unimplemented(format!(
                    "Convertion to DataFrame not yet supported: {:?}",
                    self
                )))
            }
        }
    }
}
