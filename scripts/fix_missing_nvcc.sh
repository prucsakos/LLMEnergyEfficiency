#!/bin/bash

export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda

# Append to ~/.bashrc for persistence
echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc

echo "CUDA environment variables have been set and added to ~/.bashrc"
