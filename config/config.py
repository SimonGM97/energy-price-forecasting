import os
import sys
import yaml
from typing import Dict


# Load config file
try:
    with open(os.path.join("config", "config.yaml")) as file:
        config: Dict[str, Dict[str, str]] = yaml.load(file, Loader=yaml.FullLoader)
except:
    # Remove the last part of the directory
    parent_dir = os.path.dirname(os.getcwd())
    os.chdir(parent_dir)

    with open(os.path.join("config", "config.yaml")) as file:
        config: Dict[str, Dict[str, str]] = yaml.load(file, Loader=yaml.FullLoader)

# Export S3 vars
s3_vars: dict = config.get('S3')

BUCKET: str = s3_vars.get('bucket')
REGION: str = s3_vars.get('region')