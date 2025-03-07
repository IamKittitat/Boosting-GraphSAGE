import yaml

def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)