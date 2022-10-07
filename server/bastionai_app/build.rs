fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("../protos/remote_torch.proto")?;
    tonic_build::compile_protos("../protos/attestation.proto")?;

    Ok(())
}
