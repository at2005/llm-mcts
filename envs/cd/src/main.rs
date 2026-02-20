use rand::RngExt;
use rand::seq::{IteratorRandom, SliceRandom};
use rayon::prelude::*;
use serde_json::json;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(Clone)]
pub enum Action {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl Action {
    pub fn to_string(&self) -> String {
        match self {
            Action::Add => "+".to_string(),
            Action::Subtract => "-".to_string(),
            Action::Multiply => "*".to_string(),
            Action::Divide => "/".to_string(),
        }
    }
}

#[derive(Clone)]
pub struct Step {
    pub a: u32,
    pub b: u32,
    pub op: Action,
    pub result: u32,
}

pub fn create_countdown_target(nums: Vec<u32>) -> (u32, Vec<Step>) {
    let mut rng = rand::rng();
    let mut steps = Vec::new();
    let operations = vec![
        Action::Add,
        Action::Subtract,
        Action::Multiply,
        Action::Divide,
    ];
    let mut nums = nums;

    while nums.len() > 1 {
        nums.shuffle(&mut rng);
        let a = nums.pop().unwrap();
        let b = nums.pop().unwrap();

        let op = operations.clone().into_iter().choose(&mut rng).unwrap();

        match op {
            Action::Add => match a.checked_add(b) {
                Some(v) => {
                    nums.push(v);
                    steps.push(Step {
                        a,
                        b,
                        op: Action::Add,
                        result: v,
                    });
                }
                None => {
                    nums.push(a);
                    nums.push(b);
                    continue;
                }
            },
            Action::Subtract => {
                if a <= b {
                    nums.push(a);
                    nums.push(b);
                    continue;
                }
                nums.push(a - b);
                steps.push(Step {
                    a,
                    b,
                    op: Action::Subtract,
                    result: a - b,
                });
            }

            Action::Multiply => {
                if a == 0 || b == 0 || a == 1 || b == 1 {
                    nums.push(a);
                    nums.push(b);
                    continue;
                }
                match a.checked_mul(b) {
                    Some(v) => {
                        nums.push(v);
                        steps.push(Step {
                            a,
                            b,
                            op: Action::Multiply,
                            result: v,
                        });
                    }
                    None => {
                        nums.push(a);
                        nums.push(b);
                        continue;
                    }
                }
            }

            Action::Divide => {
                if b == 0 || a % b != 0 {
                    nums.push(a);
                    nums.push(b);
                    continue;
                }
                nums.push(a / b);
                steps.push(Step {
                    a,
                    b,
                    op: Action::Divide,
                    result: a / b,
                });
            }
        }
    }

    (nums.pop().unwrap(), steps)
}

pub fn generate_sample(nums: Vec<u32>) -> (u32, Vec<Step>) {
    const NUM_STEPS: u32 = 10_000;
    let mut trajectories: HashMap<u32, Vec<Step>> = HashMap::new();
    let mut freqs = HashMap::new();

    let samples: Vec<(u32, Vec<Step>)> = (0..NUM_STEPS)
        .into_par_iter()
        .map(|_| create_countdown_target(nums.clone()))
        .collect();

    for (target, steps) in samples {
        trajectories.entry(target).or_insert(steps);
        let entry = freqs.entry(target).or_insert(0);
        *entry += 1;
    }

    let mut min_freq = u32::MAX;
    let mut min_target = u32::MAX;
    for (target, freq) in freqs {
        if freq < min_freq {
            min_freq = freq;
            min_target = target;
        }
    }

    let steps: Vec<Step> = trajectories.get(&min_target).unwrap().clone();
    return (min_target, steps);
}

pub fn create_dataset(output_path: &str, num_samples: u32) {
    let json_file = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(json_file);

    let lines: Vec<String> = (0..num_samples)
        .into_par_iter()
        .map(|_| {
            let nums = generate_input_sequence(1, 100, 6);
            let (target, steps) = generate_sample(nums.clone());
            let data = json!({
                "target": target,
                "steps": steps.iter().map(|step| json!({
                    "a": step.a,
                    "b": step.b,
                    "op": step.op.to_string(),
                    "result": step.result
                })).collect::<Vec<_>>()
            });
            let mut line = data.to_string();
            line.push('\n');
            line
        })
        .collect();

    for line in lines {
        writer.write_all(line.as_bytes()).unwrap();
    }
    writer.flush().unwrap();
}

pub fn generate_input_sequence(lb: u32, ub: u32, n: u32) -> Vec<u32> {
    let mut rng = rand::rng();
    let input_sequence = (0..n)
        .map(|_| rng.random_range(lb..=ub))
        .collect::<Vec<u32>>();
    return input_sequence;
}

fn main() {
    let mut output_path = String::from("dataset.json");
    let mut num_samples: u32 = 10_000;

    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" | "-o" => {
                if let Some(path) = args.next() {
                    output_path = path;
                } else {
                    eprintln!("--output requires a value");
                    std::process::exit(1);
                }
            }
            "--samples" | "--count" | "-n" => {
                if let Some(v) = args.next() {
                    match v.parse::<u32>() {
                        Ok(n) => num_samples = n,
                        Err(_) => {
                            eprintln!("--samples expects a positive integer, got '{}'", v);
                            std::process::exit(1);
                        }
                    }
                } else {
                    eprintln!("--samples expects a value");
                    std::process::exit(1);
                }
            }
            "--help" | "-h" => {
                println!("Usage: cargo run -- [--output <file>] [--samples <n>]");
                return;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                eprintln!("Usage: cargo run -- [--output <file>] [--samples <n>]");
                std::process::exit(1);
            }
        }
    }

    create_dataset(&output_path, num_samples);
}
