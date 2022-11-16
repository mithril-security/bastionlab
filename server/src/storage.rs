use polars::prelude::DataFrame;

#[derive(Debug, Default, Clone)]
pub struct Owner {
    address: String,
    pubKey: String,
}

#[derive(Debug, Default, Clone)]
pub struct Artifact {
    inner: DataFrame,
    pub fetchable: bool,
    owner: Owner,
}

impl Artifact {
    pub fn new(df: DataFrame, owner: Owner) -> Self {
        Artifact {
            inner: df,
            fetchable: false,
            owner,
        }
    }
    pub fn get_inner(&self) -> DataFrame {
        self.inner.clone()
    }

    pub fn make_fetchable(&mut self) {
        self.fetchable = true;
    }
}
