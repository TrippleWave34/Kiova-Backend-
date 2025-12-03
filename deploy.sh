#!/bin/bash

# Configuration
REGISTRY="kiovaregistry"
IMAGE="kiova-backend"
TAG="v$(date +%s)" # Generates a unique version number based on time
APP_NAME="kiova-api"
RESOURCE_GROUP="kiova-backend"

echo "🚀 Starting Deployment..."

# 1. Login to Azure Container Registry
echo "🔑 Logging into Registry..."
az acr login --name $REGISTRY

# 2. Build Image Locally
echo "🔨 Building Image ($TAG)..."
docker build -t $REGISTRY.azurecr.io/$IMAGE:$TAG .

# 3. Push Image to Cloud
echo "☁️ Pushing to Azure..."
docker push $REGISTRY.azurecr.io/$IMAGE:$TAG

# 4. Update the Server to use the new image
echo "🔄 Updating Container App..."
az containerapp update \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --image $REGISTRY.azurecr.io/$IMAGE:$TAG

echo "✅ Deployment Complete! Revision $TAG is live."