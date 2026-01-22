fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/inference.proto");

    tonic_prost_build::configure()
        .btree_map(".")
        .build_server(true)
        .compile_protos(&["proto/inference.proto"], &["proto"])?;
    Ok(())
}
