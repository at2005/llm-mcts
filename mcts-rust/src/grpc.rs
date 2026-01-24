use std::collections::BTreeMap;

use anyhow::Result;
use tonic::{Request, transport::Channel};

pub use crate::inference;
use inference::InferenceRequest;
use inference::inference_client::InferenceClient;

pub async fn get_client() -> Result<InferenceClient<Channel>> {
    let host = std::env::var("INFERENCE_HOST").unwrap_or_else(|_| "http://[::1]:50051".to_string());
    let client = InferenceClient::connect(host).await?;
    Ok(client)
}

pub async fn policy_value_head(
    client: &mut InferenceClient<Channel>,
    state: &[u64],
    topk: usize,
) -> Result<(BTreeMap<u32, f32>, f32)> {
    let mut request = Request::new(InferenceRequest {
        state: state.to_vec(),
        topk: topk as u32,
    });
    request.set_timeout(std::time::Duration::from_secs(120));
    let response = client.infer(request).await?.into_inner();
    Ok((response.prior, response.value))
}
