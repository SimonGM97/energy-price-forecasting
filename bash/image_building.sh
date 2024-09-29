#!/bin/bash

# chmod +x ./bash/image_building.sh
# ./bash/image_building.sh

# Set variables
ECR_REPOSITORY_NAME=energy-price-forecasting-ecr
ECR_REPOSITORY_URI=097866913509.dkr.ecr.sa-east-1.amazonaws.com
REGION=sa-east-1
VERSION=v1.0.0

# Clean running containers
docker rm -f $(docker ps -aq)

# Clean local images
docker rmi -f $(docker images -q)

# Make script executable
chmod +x ./scripts/inference/inference.py

# Build Docker image
docker build -t pricing_inference_image:${VERSION} -f docker/Dockerfile .

# Build lambda Docker image
docker build --platform linux/amd64 -t lambda_pricing_inference_image:${VERSION} -f docker/Dockerfile .

# Log-in to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin ${ECR_REPOSITORY_URI}

# Tag lambda docker image
docker tag lambda_pricing_inference_image:${VERSION} ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_pricing_inference_image_${VERSION}

# Push image to AWS ECR repository
docker push ${ECR_REPOSITORY_URI}/${ECR_REPOSITORY_NAME}:lambda_pricing_inference_image_${VERSION}

# Delete untagged images from ECR
IMAGES_TO_DELETE=$( aws ecr list-images --region $REGION --repository-name ${ECR_REPOSITORY_NAME} --filter "tagStatus=UNTAGGED" --query 'imageIds[*]' --output json )
aws ecr batch-delete-image --region $REGION --repository-name ${ECR_REPOSITORY_NAME} --image-ids "$IMAGES_TO_DELETE" || true

# List AWS ECR images
# aws ecr list-images --region $REGION --repository-name ${ECR_REPOSITORY_NAME}