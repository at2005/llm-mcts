// loads config from configs/config.json
use anyhow::Result;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize, Clone)]
pub struct ExperimentConfig {
    pub topk: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub inference_batch_size: u32,
    pub training_batch_size: u32,
    pub inference_max_wait_ms: u32,
    pub num_inference_gpus: u32,
    pub num_training_gpus: u32,
    pub inference_base_port: u32,
    pub redis_host: String,
    pub redis_port: u32,
    pub redis_db: u32,
    pub max_new_tokens: u32,
    pub num_workers_per_prompt: u32,
    pub num_worker_groups: u32,
    pub max_mcts_iterations: u32,
    pub max_samples_processed_per_worker: u32,
    pub c_puct: f32,
    pub virtual_loss: u32,
    pub eos_token_id: u64,
    pub c_value_loss: f32,
    pub c_policy_loss: f32,
    pub training_max_wait_ms: u32,
    pub training_max_steps: u32,
    pub value_head_path: String,
    pub policy_head_path: String,
    pub learning_rate: f32,
    pub ppo_epsilon: f32,
    pub num_ppo_inner_steps: u32,
}

pub fn load_config() -> Result<ExperimentConfig> {
    let current_dir = std::env::current_dir()?;
    let parent_dir = current_dir.parent().unwrap();
    let config = fs::read_to_string(parent_dir.join("configs/config.json"))?;
    let config: ExperimentConfig = serde_json::from_str(&config)?;
    Ok(config)
}
