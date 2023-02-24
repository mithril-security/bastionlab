use std::fmt::{Debug, Display};

use ndarray::{Array, ArrayView, Axis, Dim, IxDynImpl};
use polars::{export::arrow::types::PrimitiveType, prelude::DataFrame};
use tonic::Status;

use crate::common_conversions::{ndarray_to_df, to_status_error};

/*
   Introduces two fixes:
       - One macro for split.
       - Another for stacking
   This update removes the duplicated methods in the ArrayStore variant matching logic
*/
macro_rules! splitter {
    ($array:ident, $variant:tt,$ratios:ident, $inner_type:ty) => {{
        let (left, right) = split::<$inner_type>($array, $ratios);
        (ArrayStore::$variant(left), ArrayStore::$variant(right))
    }};
}

macro_rules! stacker {
    ($axis:ident, $arrays:ident, $variant:tt, $inner_type:ty) => {{
        let res = stack::<$inner_type>(
            $axis,
            &$arrays
                .iter()
                .map(|v| match v {
                    ArrayStore::$variant(a) => Ok(a.view()),
                    _ => {
                        return Err(Status::aborted(
                            "DataTypes for all columns should be the same",
                        ));
                    }
                })
                .collect::<Vec<_>>()[..],
        );
        ArrayStore::$variant(res?)
    }};
}

#[derive(Clone)]
pub enum ArrayStore {
    AxdynI64(Array<i64, Dim<IxDynImpl>>),
    AxdynU64(Array<u64, Dim<IxDynImpl>>),
    AxdynU32(Array<u32, Dim<IxDynImpl>>),
    AxdynF64(Array<f64, Dim<IxDynImpl>>),
    AxdynF32(Array<f32, Dim<IxDynImpl>>),
    AxdynI32(Array<i32, Dim<IxDynImpl>>),
    AxdynI16(Array<i16, Dim<IxDynImpl>>),
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
            ArrayStore::AxdynU64(a) => Self::AxdynU64(shuffle::<u64>(a, indices)),
            ArrayStore::AxdynU32(a) => Self::AxdynU32(shuffle::<u32>(a, indices)),
        }
    }

    pub fn split(&self, ratios: (f64, f64)) -> (Self, Self) {
        /*
            Arrays could be split on a several axes but in this implementation, we are
            only splitting on the Oth Axis (row-wise).
        */

        /*
           UPDATE: Replaces repeated pattern with macro to simplify implementation
        */
        match self {
            ArrayStore::AxdynI64(a) => splitter!(a, AxdynI64, ratios, i64),
            ArrayStore::AxdynI32(a) => splitter!(a, AxdynI32, ratios, i32),
            ArrayStore::AxdynF64(a) => splitter!(a, AxdynF64, ratios, f64),
            ArrayStore::AxdynF32(a) => splitter!(a, AxdynF32, ratios, f32),
            ArrayStore::AxdynI16(a) => splitter!(a, AxdynI16, ratios, i16),
            ArrayStore::AxdynU32(a) => splitter!(a, AxdynU32, ratios, u32),
            ArrayStore::AxdynU64(a) => splitter!(a, AxdynU64, ratios, u64),
        }
    }
    pub fn stack(axis: Axis, arrays: &[ArrayStore]) -> Result<ArrayStore, Status> {
        let first = arrays
            .get(0)
            .ok_or(Status::failed_precondition("Could not stack empty array"))?;

        let res = match first {
            ArrayStore::AxdynI64(_) => stacker!(axis, arrays, AxdynI64, i64),
            ArrayStore::AxdynI32(_) => stacker!(axis, arrays, AxdynI32, i32),
            ArrayStore::AxdynF64(_) => stacker!(axis, arrays, AxdynF64, f64),
            ArrayStore::AxdynF32(_) => stacker!(axis, arrays, AxdynF32, f32),
            ArrayStore::AxdynI16(_) => stacker!(axis, arrays, AxdynI16, i16),
            ArrayStore::AxdynU64(_) => stacker!(axis, arrays, AxdynU64, u64),
            ArrayStore::AxdynU32(_) => stacker!(axis, arrays, AxdynU32, u32),
        };

        Ok(res)
    }

    pub fn to_dataframe(&self, col_names: Vec<String>) -> Result<DataFrame, Status> {
        match self {
            ArrayStore::AxdynI64(a) => ndarray_to_df::<i64, Dim<IxDynImpl>>(a, col_names),
            ArrayStore::AxdynF64(a) => ndarray_to_df::<f64, Dim<IxDynImpl>>(a, col_names),
            ArrayStore::AxdynF32(a) => ndarray_to_df::<f32, Dim<IxDynImpl>>(a, col_names),
            ArrayStore::AxdynI32(a) => ndarray_to_df::<i32, Dim<IxDynImpl>>(a, col_names),
            ArrayStore::AxdynU32(a) => ndarray_to_df::<u32, Dim<IxDynImpl>>(a, col_names),
            ArrayStore::AxdynU64(a) => ndarray_to_df::<u64, Dim<IxDynImpl>>(a, col_names),
            _ => {
                return Err(Status::unimplemented(format!(
                    "Conversion to DataFrame not yet supported: {:#?}",
                    self
                )))
            }
        }
    }

    /*
       Update:
           - a cast method to easily cast types into another.
            This is useful especially for ML training when we receive `AxdynI{64, 32}` and
            the algorithm only accept the `AxdynU{64, 32}`.
    */
    pub fn cast(&self, dtype: ArrayStoreType) -> Result<Self, Status> {
        macro_rules! array_creator {
            ($slice:expr, $shape:ident, $to_fn:tt,  $res_variant:tt) => {{
                let mut transformed = vec![];
                for v in $slice.into_iter() {
                    let v = match v.$to_fn() {
                        Some(v) => v,
                        None => {
                            return Err(Status::internal("Failed to convert value into I64"));
                        }
                    };
                    transformed.push(v);
                }
                let array = ArrayBase::from_shape_vec($shape.clone(), transformed)
                    .map_err(|e| {
                        Status::internal(format!("Could create array from {:?}: {e}", $shape))
                    })?
                    .into_dyn();
                Ok(ArrayStore::$res_variant(array))
            }};
        }
        macro_rules! caster {
            ($array:expr, $prim_ty:expr) => {{
                use ndarray::ArrayBase;
                use polars::export::num::ToPrimitive;

                let shape = $array.shape().to_vec();
                match $array.as_slice() {
                    Some(s) => match $prim_ty {
                        PrimitiveType::Int64 => array_creator!(s, shape, to_i64, AxdynI64),
                        PrimitiveType::Int32 => array_creator!(s, shape, to_i32, AxdynI32),
                        PrimitiveType::Float32 => array_creator!(s, shape, to_f32, AxdynF32),
                        PrimitiveType::Float64 => array_creator!(s, shape, to_f64, AxdynF64),
                        PrimitiveType::UInt64 => array_creator!(s, shape, to_u64, AxdynU64),
                        PrimitiveType::UInt32 => array_creator!(s, shape, to_u32, AxdynU32),
                        PrimitiveType::Int16 => array_creator!(s, shape, to_i16, AxdynI16),
                        _ => {
                            return Err(Status::failed_precondition(format!(
                                "Unsupported Primitive Type: {dtype:?}"
                            )));
                        }
                    },
                    None => {
                        return Err(Status::aborted("Could not convert array into slice"));
                    }
                }
            }};
        }
        let primitive_type = match dtype {
            ArrayStoreType::Int64 => PrimitiveType::Int64,
            ArrayStoreType::UInt64 => PrimitiveType::UInt64,
            ArrayStoreType::Int32 => PrimitiveType::Int32,
            ArrayStoreType::UInt32 => PrimitiveType::UInt32,
            ArrayStoreType::Float64 => PrimitiveType::Float64,
            ArrayStoreType::Float32 => PrimitiveType::Float32,
            ArrayStoreType::Int16 => PrimitiveType::Int16,
        };

        match self {
            ArrayStore::AxdynF32(a) => caster!(a, primitive_type),
            ArrayStore::AxdynF64(a) => caster!(a, primitive_type),
            ArrayStore::AxdynI32(a) => caster!(a, primitive_type),
            ArrayStore::AxdynI64(a) => caster!(a, primitive_type),
            ArrayStore::AxdynU32(a) => caster!(a, primitive_type),
            ArrayStore::AxdynU64(a) => caster!(a, primitive_type),
            ArrayStore::AxdynI16(a) => caster!(a, primitive_type),
        }
    }

    pub fn formatter(&self) -> String {
        let dtype = match self {
            ArrayStore::AxdynI64(_) => ArrayStoreType::Int64,
            ArrayStore::AxdynU64(_) => ArrayStoreType::UInt64,
            ArrayStore::AxdynU32(_) => ArrayStoreType::UInt32,
            ArrayStore::AxdynF64(_) => ArrayStoreType::Float64,
            ArrayStore::AxdynF32(_) => ArrayStoreType::Float32,
            ArrayStore::AxdynI32(_) => ArrayStoreType::Int32,
            ArrayStore::AxdynI16(_) => ArrayStoreType::Int16,
        };
        format!(
            "ArrayStore<shape=[{:?}, {:?}], dtype={:?}>",
            self.height(),
            self.width(),
            dtype,
        )
    }
}

/// This impl is used as a fix to disallow leaking data through error logging
impl Display for ArrayStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.formatter(),)
    }
}

impl Debug for ArrayStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.formatter(),)
    }
}

/// ArrayStore's internal representation for Primitive Types which is used alongside num-traits PrimitiveTypes to
/// match on types used in cast and debug and display printing.
#[derive(Debug)]
pub enum ArrayStoreType {
    Float64,
    Float32,
    UInt64,
    Int64,
    UInt32,
    Int32,
    Int16,
}
