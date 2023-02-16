pub mod converter;

pub mod conversion_proto {
    tonic::include_proto!("bastionlab_conversion");
}

pub mod bastionlab {
    tonic::include_proto!("bastionlab");
}
