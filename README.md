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
3. Clone the `energy-price-forecasting` Github repository:
```bash
git clone https://github.com/SimonGM97/energy-price-forecasting.git
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
  - *Note that this command will install the dependencies, specified in `requirements.txt`.*