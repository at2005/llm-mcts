#!/bin/bash

sudo apt update

# install cargo/rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# install uv
curl -fsSL https://astral.sh/uv/install.sh | sh

# install redis 
sudo apt install -y redis-server
sudo systemctl enable --now redis-server