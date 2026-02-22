use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::net::TcpStream;
use tonic::{Request, Response, transport::Channel};
use tracing::warn;

pub use crate::inference;
use inference::inference_client::InferenceClient;
use inference::{
    GetPromptRequest, GetPromptResponse, GraderRequest, GraderResponse, InferenceRequest,
    InferenceResponse, PriorEntry,
};
use rand::seq::IteratorRandom;

pub struct InferenceClientPool {
    clients: BTreeMap<usize, InferenceClient<Channel>>,
    infer_client_port: usize,
    get_prompt_timeout: Duration,
    grader_timeout: Duration,
    inference_timeout: Duration,
}

impl InferenceClientPool {
    pub const DEFAULT_PORT: usize = 50051;
    pub const DEFAULT_STARTUP_TIMEOUT_MS: u64 = 120_000;
    pub const STARTUP_RETRY_INTERVAL_MS: u64 = 250;

    pub async fn new(num_servers: usize, base_port: usize, start_rank: usize) -> Result<Self> {
        Self::new_with_timeouts_for_worker(
            num_servers,
            base_port,
            start_rank,
            0,
            Duration::from_secs(30),
            Duration::from_secs(30),
            Duration::from_secs(120),
        )
        .await
    }

    pub async fn new_with_timeouts(
        num_servers: usize,
        base_port: usize,
        start_rank: usize,
        get_prompt_timeout: Duration,
        grader_timeout: Duration,
        inference_timeout: Duration,
    ) -> Result<Self> {
        Self::new_with_timeouts_for_worker(
            num_servers,
            base_port,
            start_rank,
            0,
            get_prompt_timeout,
            grader_timeout,
            inference_timeout,
        )
        .await
    }

    pub async fn new_with_timeouts_for_worker(
        num_servers: usize,
        base_port: usize,
        start_rank: usize,
        worker_pool_id: u32,
        get_prompt_timeout: Duration,
        grader_timeout: Duration,
        inference_timeout: Duration,
    ) -> Result<Self> {
        let mut clients = BTreeMap::new();
        for i in 0..num_servers {
            let port = base_port + start_rank + i;
            let client = Self::get_client(port).await?;
            clients.insert(port, client);
        }
        if clients.is_empty() {
            anyhow::bail!("no inference clients available");
        }
        let infer_client_idx = (worker_pool_id as usize) % clients.len();
        let infer_client_port = *clients
            .keys()
            .nth(infer_client_idx)
            .expect("infer client index not found");

        Ok(Self {
            clients,
            infer_client_port,
            get_prompt_timeout,
            grader_timeout,
            inference_timeout,
        })
    }

    pub async fn get_client(port: usize) -> Result<InferenceClient<Channel>> {
        let host = format!("http://127.0.0.1:{}", port);
        let client = InferenceClient::connect(host).await?;
        Ok(client)
    }

    pub async fn wait_for_servers(
        num_servers: usize,
        base_port: usize,
        start_rank: usize,
        max_wait: Duration,
    ) -> Result<()> {
        let start = Instant::now();
        loop {
            let mut ready = true;
            for i in 0..num_servers {
                let addr = format!("127.0.0.1:{}", base_port + start_rank + i);
                if TcpStream::connect(&addr).await.is_err() {
                    ready = false;
                    break;
                }
            }

            if ready {
                return Ok(());
            }

            if start.elapsed() >= max_wait {
                anyhow::bail!(
                    "Timed out waiting for inference servers to come online after {}ms",
                    max_wait.as_millis()
                );
            }

            warn!(
                "Some inference servers not online yet. Retrying in {}ms...",
                Self::STARTUP_RETRY_INTERVAL_MS
            );
            tokio::time::sleep(Duration::from_millis(Self::STARTUP_RETRY_INTERVAL_MS)).await;
        }
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

    pub fn get_sticky_infer_client(&self) -> Result<InferenceClient<Channel>> {
        let client = self
            .clients
            .get(&self.infer_client_port)
            .expect("sticky infer client not found")
            .clone();
        Ok(client)
    }

    pub async fn send_request(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceResponse>> {
        let mut client = self.get_sticky_infer_client()?;
        let response = client.infer(request).await?;
        Ok(response)
    }

    pub async fn send_get_prompt_request(
        &self,
        worker_pool_id: u32,
    ) -> Result<Response<GetPromptResponse>> {
        let mut client = self.get_random_client().await?;
        let mut request = Request::new(GetPromptRequest { worker_pool_id });
        request.set_timeout(self.get_prompt_timeout);
        let response = client.get_prompt(request).await?;
        Ok(response)
    }

    pub async fn send_grader_request(
        &self,
        episode_id: u32,
        prompt_id: u32,
        state: Vec<u64>,
    ) -> Result<Response<GraderResponse>> {
        let mut request = Request::new(GraderRequest {
            episode_id,
            prompt_id,
            state,
        });
        request.set_timeout(self.grader_timeout);
        let mut client = self.get_random_client().await?;
        let response = client.grader(request).await?;
        Ok(response)
    }

    pub async fn policy_value_head(&self, state: &[u64]) -> Result<(Vec<PriorEntry>, f32)> {
        let mut request = Request::new(InferenceRequest {
            state: state.to_vec(),
        });
        request.set_timeout(self.inference_timeout);
        let response = self.send_request(request).await?.into_inner();
        Ok((response.priors, response.value))
    }
}
