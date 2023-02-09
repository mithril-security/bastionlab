use std::sync::{Arc, Mutex};

use bastionlab_common::common_conversions::*;
use bastionlab_polars::BastionLabPolars;
use bastionlab_torch::BastionLabTorch;
use tonic::{Request, Response, Status};

use crate::conversion_proto::{conversion_service_server::ConversionService, ToTensor};

use crate::bastionlab::Reference;
pub struct Converter {
    torch: Arc<BastionLabTorch>,
    polars: Arc<BastionLabPolars>,
}

impl Converter {
    pub fn new(torch: Arc<BastionLabTorch>, polars: Arc<BastionLabPolars>) -> Self {
        Self { torch, polars }
    }
}

#[tonic::async_trait]
impl ConversionService for Converter {
    async fn conv_to_tensor(
        &self,
        request: Request<ToTensor>,
    ) -> Result<Response<Reference>, Status> {
        let identifier = &request.get_ref().identifier;

        let df = self.polars.get_df_unchecked(&identifier)?;

        let mut tensor = df_to_tensor(&df)?;
        // Important Note:
        // DF -> Tensor always returns a 2-d tensor.
        // For a single column DF, we will have to reshape Tensor as 1-d.

        if df.width() == 1 {
            tensor = tensor.squeeze();
        }

        let (_, tensor_ref) = self.torch.insert_tensor(Arc::new(Mutex::new(tensor)));

        // Here, a dataframe has been converted and can be cleared from the Polars storage.
        self.polars.delete_dfs(identifier)?;

        let tensor_ref = Reference {
            identifier: tensor_ref.identifier,
            meta: tensor_ref.meta,
            ..Default::default()
        };
        Ok(Response::new(tensor_ref))
    }
}
