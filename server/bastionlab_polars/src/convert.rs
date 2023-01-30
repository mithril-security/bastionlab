use tonic::Status;

use crate::errors::BastionLabPolarsError;

pub struct OptionBoxTryFrom<T>(pub Option<Box<T>>);

pub struct VecTryFrom<T>(pub Vec<T>);

impl<T, U, E> TryFrom<Vec<U>> for VecTryFrom<T>
where
    T: TryFrom<U, Error = E>,
    BastionLabPolarsError: From<E>,
{
    type Error = BastionLabPolarsError;

    fn try_from(value: Vec<U>) -> Result<Self, Self::Error> {
        let mut res = Vec::with_capacity(value.len());
        for e in value {
            res.push(e.try_into()?);
        }
        Ok(VecTryFrom(res))
    }
}

impl<T, U, E> TryFrom<VecTryFrom<U>> for Vec<T>
where
    T: TryFrom<U, Error = E>,
    BastionLabPolarsError: From<E>,
{
    type Error = BastionLabPolarsError;

    fn try_from(value: VecTryFrom<U>) -> Result<Self, Self::Error> {
        let mut res = Vec::with_capacity(value.0.len());
        for e in value.0 {
            res.push(e.try_into()?);
        }
        Ok(res)
    }
}

impl<T, U, E> TryFrom<Option<Box<U>>> for OptionBoxTryFrom<T>
where
    T: TryFrom<U, Error = E>,
    BastionLabPolarsError: From<E>,
{
    type Error = BastionLabPolarsError;

    #[inline]
    fn try_from(value: Option<Box<U>>) -> Result<Self, Self::Error> {
        match value {
            Some(x) => Ok(OptionBoxTryFrom(Some(Box::new((*x).try_into()?)))),
            None => Ok(OptionBoxTryFrom(None)),
        }
    }
}

impl<T, U, E> TryFrom<OptionBoxTryFrom<U>> for Option<Box<T>>
where
    T: TryFrom<U, Error = E>,
    BastionLabPolarsError: From<E>,
{
    type Error = BastionLabPolarsError;

    #[inline]
    fn try_from(value: OptionBoxTryFrom<U>) -> Result<Self, Self::Error> {
        match value.0 {
            Some(x) => Ok(Some(Box::new((*x).try_into()?))),
            None => Ok(None),
        }
    }
}

pub fn polars_to_status_error<T, E: std::error::Error>(err: Result<T, E>) -> Result<T, Status> {
    err.map_err(|e| Status::internal(format!("Could not perform operation: {e}",)))
}
