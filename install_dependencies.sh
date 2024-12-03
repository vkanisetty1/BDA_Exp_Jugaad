#!/bin/bash

# Script to install project dependencies

echo "Updating package list..."
sudo apt-get update -y || echo "Package update failed. Skipping..."

echo "Installing Python3 and pip (if not already installed)..."
sudo apt-get install -y python3 python3-pip || echo "Python3 or pip installation failed. Skipping..."

echo "Installing required Python libraries..."
pip3 install --upgrade pip || echo "Pip upgrade failed."
pip3 install langchain sentence-transformers faiss-cpu transformers streamlit PyPDF2 || {
    echo "Error occurred while installing dependencies."
    exit 1
}

echo "Dependencies installed successfully!"
