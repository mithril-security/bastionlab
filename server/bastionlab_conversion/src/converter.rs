use std::sync::Arc;

use bastionlab_common::common_conversions::*;
use bastionlab_common::session::SessionManager;
use bastionlab_polars::BastionLabPolars;
use bastionlab_torch::BastionLabTorch;
use prost::Message;
use tonic::{Request, Response, Status};

use crate::conversion_proto::{conversion_service_server::ConversionService, ToTensor};

use crate::bastionlab::{Reference, TensorMetaData};
pub struct Converter {
    torch: Arc<BastionLabTorch>,
    polars: Arc<BastionLabPolars>,
    sess_manager: Arc<SessionManager>,
}

impl Converter {
    pub fn new(
        torch: Arc<BastionLabTorch>,
        polars: Arc<BastionLabPolars>,
        sess_manager: Arc<SessionManager>,
    ) -> Self {
        Self {
            torch,
            polars,
            sess_manager,
        }
    }
}

#[tonic::async_trait]
impl ConversionService for Converter {
    async fn conv_to_tensor(
        &self,
        request: Request<ToTensor>,
    ) -> Result<Response<Reference>, Status> {
        self.sess_manager.verify_request(&request)?;
        let identifier = &request.get_ref().identifier;

        let df = self.polars.get_df_unchecked(&identifier)?;

        let tensor = {
            let cols = df.get_column_names();
            if cols.len() == 1 {
                let series = to_status_error(df.column(cols[0]))?;
                let data = series_to_tensor(series)?;

                data
            } else {
                let tensor = df_to_tensor(&df)?;
                tensor
            }
        };
        let meta = TensorMetaData {
            input_dtype: vec![format!("{:?}", tensor.kind())],
            input_shape: tensor.size(),
        };

        let tensor_id = self.torch.insert_tensor(tensor);

        // Here, a dataframe has been converted and can be cleared from the Polars storage.
        self.polars.delete_dfs(identifier)?;

        Ok(Response::new(Reference {
            identifier: tensor_id,
            name: String::new(),
            description: String::new(),
            meta: meta.encode_to_vec(),
        }))
    }
}
