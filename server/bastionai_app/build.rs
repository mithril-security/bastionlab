fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../protos/remote_torch.proto");
    tonic_build::compile_protos("../../protos/remote_torch.proto")?;

    Ok(())
}
