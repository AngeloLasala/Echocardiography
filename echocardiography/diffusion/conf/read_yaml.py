import yaml

# Specify the path to your YAML file
yaml_file_path = 'mnist.yaml'

# Read the YAML file
with open(yaml_file_path, 'r') as file:
    config_data = yaml.safe_load(file)

# Access the configuration parameters
autoencoder_config = config_data.get('autoencoder_config', {})
print(autoencoder_config)