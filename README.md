# energy-price-forecasting
Repositorio en donde se desarrollará el trabajo práctico para la materia Implementación de Aplicaciones de Aprendizaje Automático en la Nube - ITBA

&nbsp;
# Table of Contents

- [Installation](#installation)
- [Usage](#usage)

&nbsp;
# Installation

1. Install the AWS CLI v2 (if it's not already installed)
```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg ./AWSCLIV2.pkg -target /
```
2. Set up the IAM credentials using aws configure:
```bash
aws configure
```
```
AWS Access Key ID: AWS_ACCESS_KEY_ID
AWS Secret Access Key: AWS_SECRET_ACCESS_KEY
Default region name: sa-east-1
Default output format: json
```
3. Clone the `PyTradeX` CodeCommit repository:
```bash
git config --global credential.helper '!aws codecommit credential-helper $@'
git config --global credential.UseHttpPath true
git config --global user.email "simongmorillo1@gmail.com"
git config --global user.name "simon.garcia.morillo"
git clone https://git-codecommit.sa-east-1.amazonaws.com/v1/repos/PyTradeX
```
4. Create & activate python virtual environment:
```bash
python -m venv .price_forecasting_env
source .price_forecasting_env/bin/activate
```
5. Install requirements
```bash
pip install -r requirements.txt
```
  - *Note that this command will also install the dependencies, specified in `requirements.txt`.*
6. Install & run the [Docker Desktop](https://docs.docker.com/engine/install/) application (if it's not already installed). 
7. Set the `ECR` environment variables to pull images from the `pytradex-ecr` ECR repository:
```bash
export ECR_REPOSITORY_NAME=pytradex-ecr
export ECR_REPOSITORY_URI=097866913509.dkr.ecr.sa-east-1.amazonaws.com
export REGION=sa-east-1
```
8. Pull the docker images stored in the `ed-ml-docker` dockerhub repository:
```
chmod +x ./scripts/bash/image_pulling.sh
./scripts/bash/image_pulling.sh
```
- Note that this will pull the following docker images:
  - `data_processing_image_v1.0.0`
  - `model_tuning_image_v1.0.0`
  - `model_updating_image_v1.0.0`
  - `model_serving_image_v1.0.0`
  - `inference_image_v1.0.0`
  - `run_app_image_v1.0.0`