#!/bin/bash

# chmod +x ./bash/lambda_function_building.sh
# ./bash/lambda_function_building.sh

# Set variables
ECR_REPOSITORY_NAME=energy-price-forecasting-ecr
ECR_REPOSITORY_URI=097866913509.dkr.ecr.sa-east-1.amazonaws.com
LAMBDA_ROLE_ARN=arn:aws:iam::097866913509:role/lambda_role
REGION=sa-east-1
VERSION=v1.0.0

# """
# INFERENCE
# """
# Delete trading lambda function
aws lambda delete-function --function-name pricing-inference

# Create new trading lambda function
aws lambda create-function \
    --function-name pricing-inference \
    --role ${LAMBDA_ROLE_ARN} \
    --package-type Image \
    --code ImageUri=${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_pricing_inference_image_${VERSION} \
    --description "Pricing inference lambda function." \
    --architecture x86_64 \
    --memory-size 1024 \
    --timeout 120 \
    --region ${REGION} \
    --publish