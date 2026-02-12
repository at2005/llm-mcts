use crate::mcts::{Node, NodeId};
use anyhow::Result;
use reqwest::Client;
use serde_json::json;
use std::sync::atomic::Ordering;
use tracing::info;

pub struct Logger {
    http_client: reqwest::Client,
    base_url: String,
    active: bool,
    worker_id: u32,
}

impl Logger {
    pub fn new(base_url: Option<String>, worker_id: u32) -> Result<Self> {
        let base_url = base_url.unwrap_or("http://localhost:3001".to_string());
        let http_client = Client::new();
        Ok(Self {
            http_client,
            base_url,
            active: true,
            worker_id,
        })
    }

    pub fn deactivate(&mut self) {
        self.active = false;
    }

    pub fn activate(&mut self) {
        self.active = true;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub async fn get_tree(&self) -> Result<()> {
        if !self.active {
            return Ok(());
        }

        let response = self
            .http_client
            .get(format!("{}/api/tree/{}", self.base_url, self.worker_id))
            .send()
            .await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get tree: {}", response.status()));
        }

        Ok(())
    }

    pub async fn reset_tree(&self) -> Result<()> {
        if !self.active {
            return Ok(());
        }

        if self.get_tree().await.is_err() {
            info!("Tree not found, skipping reset");
            return Ok(());
        }

        let response = self
            .http_client
            .post(format!(
                "{}/api/reset-tree/{}",
                self.base_url, self.worker_id
            ))
            .send()
            .await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to reset tree: {}",
                response.status()
            ));
        }

        Ok(())
    }

    pub async fn log_node(&self, node_id: NodeId, node: &Node) -> Result<()> {
        if !self.active {
            return Ok(());
        }

        let payload = json!({
            "id": node_id,
            "parentId": node.parent.unwrap_or(usize::MAX),
            "contents": node.state.iter().copied().collect::<Vec<_>>(),
            "visits": node.visits.load(Ordering::Relaxed),
            "value": node.accumulated_value.load(Ordering::Relaxed),
            "workerId": self.worker_id,
        });

        let response = self
            .http_client
            .post(format!("{}/api/nodes", self.base_url))
            .header("Content-Type", "application/json")
            .body(payload.to_string())
            .send()
            .await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to log node: {}", response.status()));
        }

        Ok(())
    }
}
