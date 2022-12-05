use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../protos/bastionlab.proto");
    tonic_build::compile_protos("../../protos/bastionlab.proto")?;

    Ok(())
}
