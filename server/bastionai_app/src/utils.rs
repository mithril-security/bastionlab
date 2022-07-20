use crate::remote_torch::*;
use anyhow::{Context, Result};
use http::Uri;
use std::net::SocketAddr;
use std::net::ToSocketAddrs;
use tch::*;

pub fn as_u32_le(array: &[u8]) -> u32 {
    ((array[0] as u32) << 0)
        + ((array[1] as u32) << 8)
        + ((array[2] as u32) << 16)
        + ((array[3] as u32) << 24)
}

pub fn from_u32_le(size: u32) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(4);
    out.push(((size >> 24) & 0xff) as u8);
    out.push(((size >> 16) & 0xff) as u8);
    out.push(((size >> 8) & 0xff) as u8);
    out.push((size & 0xff) as u8);
    out
}

pub fn bytes_to_module(mut data: &[u8]) -> Result<tch::TrainableCModule, TchError> {
    let module: TrainableCModule =
        TrainableCModule::load_data(&mut data, nn::VarStore::new(Device::Cpu).root())?;
    Ok(module)
}

pub fn bytes_to_tensor(data: &[u8]) -> Result<Tensor, TchError> {
    let tensor = Tensor::of_slice(data);
    Ok(tensor)
}

pub fn transform_bytes<T>(
    mut data: Vec<u8>,
    func: impl Fn(&[u8]) -> Result<T, TchError>,
) -> (Vec<T>, Vec<Vec<u8>>) {
    let mut transformed: Vec<T> = Vec::new();
    let mut ret_bytes: Vec<Vec<u8>> = Vec::new();

    while data.len() > 0 {
        let len: usize = as_u32_le(&data.drain(..4).collect::<Vec<u8>>()[..]) as usize;
        let bytes = &data.drain(..len).collect::<Vec<u8>>();
        ret_bytes.push(bytes.clone());
        transformed.push(func(&bytes[..]).unwrap());
    }

    (transformed, ret_bytes)
}

pub fn get_available_objects(objects: Vec<(String, String)>) -> Vec<AvailableObject> {
    let res: Vec<AvailableObject> = objects
        .into_iter()
        .map(|(k, v)| AvailableObject {
            reference: k.to_string(),
            description: v.to_string(),
        })
        .collect::<Vec<AvailableObject>>();
    res
}

pub fn uri_to_socket(uri: &Uri) -> Result<SocketAddr> {
    uri.authority()
        .context("No authority")?
        .as_str()
        .to_socket_addrs()?
        .next()
        .context("Uri could not be converted to socket")
}

fn zeros(size: usize) -> Vec<u8> {
    std::iter::repeat(0).take(size).collect()
}

pub fn serialize_tensor(tensor: &Tensor) -> Vec<u8> {
    let capacity = tensor.numel() * tensor.f_kind().unwrap().elt_size_in_bytes();
    let mut data: Vec<u8> = zeros(capacity);
    tensor.copy_data_u8(&mut data[..], tensor.numel());
    let mut capacity_bytes = from_u32_le(capacity as u32);
    capacity_bytes.append(&mut data);

    capacity_bytes
}
