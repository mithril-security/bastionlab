use super::polars_proto::{fetch_chunk, FetchChunk, SendChunk};
use crate::prelude::*;
use crate::{DataFrameArtifact, DelayedDataFrame, FetchStatus};
use polars::prelude::*;
use ring::digest;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tonic::{Response, Status};

const CHUNK_SIZE: usize = 32 * 1024;

// TODO PERF: Do a PR on polars/pypolars to add the streaming IPC (apache flight) format to the python interface
// right now, there is only the file format which requires random access
// which means, we have to do a full copy to a buffer and we cannot parse it as we go
// also: polar's IpcStreamReader requires the underlying stream to be Seek; which is weird & does not make sense

pub async fn unserialize_dataframe(
    mut stream: tonic::Streaming<SendChunk>,
) -> Result<(DataFrameArtifact, String), Status> {
    let mut buf: Vec<u8> = Vec::new();
    let mut first = true;
    let mut policy = String::new();
    let mut sanitized_columns = Vec::new();

    let mut hasher = digest::Context::new(&digest::SHA256);

    while let Some(chunk) = stream.next().await {
        let mut chunk = chunk?;
        hasher.update(&chunk.data);
        buf.append(&mut chunk.data);
        if first {
            policy = chunk.policy;
            sanitized_columns = chunk.sanitized_columns;
            first = false;
        }
    }

    let hash = hex::encode(hasher.finish().as_ref());

    let policy = serde_json::from_str(&policy).map_err(|err| {
        Status::invalid_argument(format!("Error during the parsing of the policy: {err}"))
    })?;

    let view = std::io::Cursor::new(&buf);
    let df = polars::io::ipc::IpcReader::new(view)
        .finish()
        .map_err(|err| Status::invalid_argument(format!("Polars error: {err}")))?;

    Ok((DataFrameArtifact::new(df, policy, sanitized_columns), hash))
}

// so, to hash a dataset, this does a full serialization; that's kinda bad
pub fn hash_dataset(df: &mut DataFrame) -> Result<String, PolarsError> {
    let buf = dataframe_ser_helper(df)?;
    Ok(hex::encode(digest::digest(&digest::SHA256, &buf).as_ref()))
}

// This requires &mut because polars is kinda weird about that, but it's not mutated..
fn dataframe_ser_helper(df: &mut DataFrame) -> Result<Vec<u8>, PolarsError> {
    let mut buf = Vec::new();

    // PERF: this can be replaced by manually using arrow IPC methods, to avoid this copy
    let view = std::io::Cursor::new(&mut buf);
    polars::io::ipc::IpcWriter::new(view).finish(df)?;

    Ok(buf)
}

pub async fn serialize_delayed_dataframe(
    df: DelayedDataFrame,
) -> Response<ReceiverStream<Result<FetchChunk, Status>>> {
    let (tx, rx) = mpsc::channel(4);

    match df.fetch_status {
        FetchStatus::Pending(reason) => tx
            .send(Ok(FetchChunk {
                body: Some(fetch_chunk::Body::Pending(reason)),
            }))
            .await
            .unwrap(),
        FetchStatus::Warning(reason) => tx
            .send(Ok(FetchChunk {
                body: Some(fetch_chunk::Body::Warning(reason)),
            }))
            .await
            .unwrap(),
        _ => (),
    }

    tokio::spawn(async move {
        // important things to note about tokio channels:
        // - send() on them will block until there is space in the queue
        // - send() returns an error when the receiver has been dropped / .close() has been called on it
        //   this means that send() will return Err only when the client has "lost interest", has dropped the connection / call

        let mut df: DataFrame = match df.future.await {
            Ok(df) => df,
            Err(e) => {
                // ignore send() error: error means the channel has been closed, ie, client dropped the request.
                let _ignored = tx.send(Err(e)).await;
                return;
            }
        };

        let res = dataframe_ser_helper(&mut df)
            .map_err(|err| Status::internal(format!("Polars error: {err}"))); // this is an internal error

        let buf = match res {
            Ok(buf) => buf,
            Err(err) => {
                // ignore send() error
                let _ignored = tx.send(Err(err)).await;
                return;
            }
        };

        for chunk in buf.chunks(CHUNK_SIZE) {
            let data = FetchChunk {
                body: Some(fetch_chunk::Body::Data(chunk.into())),
            };

            if let Err(_ignored) = tx.send(Ok(data)).await {
                // we have a send() error, meaning client isnt listening anymore
                // stop the task when this is the case
                return;
            }
        }
    });

    Response::new(ReceiverStream::new(rx))
}
