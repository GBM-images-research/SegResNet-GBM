import os
import json 

def load_config(config_file):
    # Load configuration from JSON file
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    else:
        raise FileNotFoundError(f"Config file '{config_file}' not found.")

def save_config(config, config_file):
    # Save configuration to JSON file
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)