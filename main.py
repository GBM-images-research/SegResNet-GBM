import os
import json
import argparse
import torch
from data_loader import get_data_loaders
from model import get_model_and_train_components
from train import train_model
from utils import load_config

def main(config_file):
    print(f"Loading config file from: {config_file}")  
    config = load_config(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_data_loaders(config["root_dir"])
    model = get_model_and_train_components(config["train"], device)
    train_model(model, train_loader, val_loader, config["train"], config["wandb_api_key"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Convolutional Neural Network for Brain Tumor Segmentation")
    parser.add_argument("--config", default="/home/psalmon/SegResNet_Testing_Brats/SegResNet-GBM/config_file.json", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)