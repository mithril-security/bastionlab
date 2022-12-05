use polars::prelude::*;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tonic::{Response, Status};

use crate::{access_control::Policy, DataFrameArtifact, DelayedDataFrame};

use super::polars_proto::{FetchChunk, SendChunk};

pub async fn df_artifact_from_stream(
    stream: tonic::Streaming<SendChunk>,
) -> Result<DataFrameArtifact, Status> {
    let (df_bytes, policy, metadata) = unstream_data(stream).await?;
    println!("{}", policy);
    let series = df_bytes
        .iter()
        .map(|v| bincode::deserialize(&v[..]).unwrap())
        .collect::<Vec<Series>>();
    let df = DataFrame::new(series.clone())
        .map_err(|_| Status::unknown("Failed to deserialize DataFrame."))?;
    let policy: Policy = serde_json::from_str(&policy)
        .map_err(|_| Status::unknown("Failed to deserialize policy."))?;
    let blacklist: Vec<String> = serde_json::from_str(&metadata)
        .map_err(|_| Status::unknown("Failed to deserialize metadata."))?;
    Ok(DataFrameArtifact::new(df, policy, blacklist))
}

pub fn df_to_bytes(df: &DataFrame) -> Vec<Vec<u8>> {
    let series = df.get_columns();
    let series_bytes = series
        .iter()
        .map(|s| bincode::serialize(s).unwrap())
        .collect::<Vec<Vec<u8>>>();
    series_bytes
}

pub async fn unstream_data(
    mut stream: tonic::Streaming<SendChunk>,
) -> Result<(Vec<Vec<u8>>, String, String), Status> {
    let mut columns: Vec<u8> = Vec::new();
    let mut policy = String::new();
    let mut metadata = String::new();

    while let Some(chunk) = stream.next().await {
        let mut chunk = chunk?;
        columns.append(&mut chunk.data);
        policy.push_str(&chunk.policy);
        metadata.push_str(&chunk.metadata);
    }

    let pattern = b"[end]";
    let mut indexes = vec![0 as usize];
    indexes.append(
        &mut columns
            .windows(pattern.len())
            .enumerate()
            .map(
                |(i, slide): (usize, &[u8])| {
                    if slide == pattern {
                        i
                    } else {
                        usize::MIN
                    }
                },
            )
            .filter(|v| v != &usize::MIN)
            .collect::<Vec<usize>>(),
    );
    let output = indexes
        .windows(2)
        .map(|r| {
            let start;
            if r[0] == 0 {
                start = r[0];
            } else {
                start = r[0] + 5;
            }
            let end = r[1];

            columns[start..end].to_vec()
        })
        .collect::<Vec<Vec<u8>>>();
    Ok((output, policy, metadata))
}

/// Converts a raw artifact (a header and a binary object) into a stream of chunks to be sent over gRPC.
pub async fn stream_data(
    df: DelayedDataFrame,
    chunk_size: usize,
) -> Response<ReceiverStream<Result<FetchChunk, Status>>> {
    let (tx, rx) = mpsc::channel(4);
    let pattern = b"[end]";

    if let Some(reason) = df.needs_approval {
        tx.send(Ok(FetchChunk {
            data: Vec::new(),
            pending: reason,
        }))
        .await
        .unwrap(); // fix this
    }

    tokio::spawn(async move {
        let df: DataFrame = match df.future.await {
            Ok(df) => df,
            Err(e) => {
                tx.send(Err(e)).await.unwrap(); // fix this
                return;
            }
        };

        let df_bytes = df_to_bytes(&df)
            .iter_mut()
            .map(|v| {
                v.append(&mut pattern.to_vec());
                v.clone()
            })
            .flatten()
            .collect::<Vec<_>>();

        let raw_bytes: Vec<u8> = df_bytes;

        for (_, bytes) in raw_bytes.chunks(chunk_size).enumerate() {
            tx.send(Ok(FetchChunk {
                data: bytes.to_vec(),
                pending: String::new(),
            }))
            .await
            .unwrap(); // Fix this
        }
    });

    Response::new(ReceiverStream::new(rx))
}
