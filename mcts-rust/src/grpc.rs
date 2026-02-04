use std::collections::BTreeMap;

use anyhow::Result;
use tonic::{Request, Response, transport::Channel};

pub use crate::inference;
use inference::inference_client::InferenceClient;
use inference::{
    GetPromptRequest, GetPromptResponse, GraderRequest, GraderResponse, InferenceRequest,
    InferenceResponse, PriorEntry,
};
use rand::seq::IteratorRandom;

pub struct InferenceClientPool {
    clients: BTreeMap<usize, InferenceClient<Channel>>,
}

impl InferenceClientPool {
    pub const DEFAULT_PORT: usize = 50051;

    pub async fn new(num_servers: usize) -> Result<Self> {
        let mut clients = BTreeMap::new();
        let base_port = Self::DEFAULT_PORT;
        for i in 0..num_servers {
            let port = base_port + i;
            let client = Self::get_client(Some(port)).await?;
            clients.insert(port, client);
        }

        Ok(Self { clients })
    }

    pub async fn get_client(port: Option<usize>) -> Result<InferenceClient<Channel>> {
        let port = port.unwrap_or(Self::DEFAULT_PORT);
        let host = format!("http://[::1]:{}", port);
        let client = InferenceClient::connect(host).await?;
        Ok(client)
    }

    pub async fn get_random_client(&self) -> Result<InferenceClient<Channel>> {
        let random_port = self.clients.keys().choose(&mut rand::rng()).unwrap();
        let client = self
            .clients
            .get(random_port)
            .expect("client not found")
            .clone();
        Ok(client)
    }

    pub async fn send_request(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceResponse>> {
        let mut client = self.get_random_client().await?;
        let response = client.infer(request).await?;
        Ok(response)
    }

    pub async fn send_get_prompt_request(
        &self,
        worker_pool_id: u32,
    ) -> Result<Response<GetPromptResponse>> {
        let mut client = self.get_random_client().await?;
        let response = client
            .get_prompt(GetPromptRequest { worker_pool_id })
            .await?;
        Ok(response)
    }

    pub async fn send_grader_request(
        &self,
        episode_id: u32,
        prompt_id: u32,
        state: Vec<u64>,
    ) -> Result<Response<GraderResponse>> {
        let grader_request = GraderRequest {
            episode_id,
            prompt_id,
            state,
        };
        let mut client = self.get_random_client().await?;
        let response = client.grader(grader_request).await?;
        Ok(response)
    }

    pub async fn policy_value_head(&self, state: &[u64]) -> Result<(Vec<PriorEntry>, f32)> {
        let mut request = Request::new(InferenceRequest {
            state: state.to_vec(),
        });
        request.set_timeout(std::time::Duration::from_secs(120));
        let response = self.send_request(request).await?.into_inner();
        Ok((response.priors, response.value))
    }
}
