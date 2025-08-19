Churn Prediction Model - Deployment Guide
This guide provides a step-by-step process for deploying the churn prediction model to a production environment using Docker, Docker Compose, and a reverse proxy for SSL.

1. Prerequisites
Before you begin, ensure the following are installed and configured on your local machine and production server:

Docker and Docker Compose: For containerisation and orchestration.

Git: To clone the repository.

SSH Access: To connect to the production server.

SSL Certificate: For secure HTTPS communication on the production server.

Monitoring Infrastructure: A running instance of Prometheus and Grafana.

2. Environment Setup
Begin by setting up your local environment and preparing the necessary files.

# Clone the repository
git clone https://github.com/Muhammadali1029/ml-engineering-training.git
cd ml-engineering-training/churn-prediction

# Create the production environment file from the example
cp .env.example .env.production

# IMPORTANT: Edit .env.production with your specific production values,
# such as API keys and database connection strings.


3. Build and Test
Next, build the Docker image and run the tests to ensure everything is working correctly before deployment.

# Build the production Docker image
docker build -t churn-prediction:prod .

# Run the tests inside the container
docker run --rm -e PYTHONPATH=/app churn-prediction:prod pytest tests/

# Test the application locally using Docker Compose (for testing purposes)
docker-compose up -d
curl http://localhost:5001/health


4. Deployment to Production
Follow these steps to deploy the application to your production server.

# Tag the Docker image for your registry
docker tag churn-prediction:prod registry.company.com/churn-prediction:v1.0

# Push the tagged image to your container registry
docker push registry.company.com/churn-prediction:v1.0

# Log in to your production server via SSH
ssh production-server

# Pull the latest image from the registry on the server
docker pull registry.company.com/churn-prediction:v1.0

# Deploy the application using the production Docker Compose file
docker-compose -f docker-compose.prod.yml up -d


5. Configure Monitoring
Proper monitoring is crucial for a production service.

# Add this job to your Prometheus configuration (prometheus.yml) to scrape the API metrics
- job_name: 'churn-prediction'
  static_configs:
    - targets: ['churn-api:5001']


You should also import the provided Grafana dashboard and configure alerts.

Import the dashboard: monitoring/grafana-dashboard.json

Set up alerts based on the rules in: monitoring/alerts.yml

6. Set up SSL/TLS with NGINX
For secure communication, use a reverse proxy like NGINX to handle SSL/TLS.

server {
    listen 443 ssl;
    server_name api.churnprediction.company.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    location / {
        proxy_pass http://localhost:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}


7. Rollback Procedure
In case of a critical issue, you can quickly roll back to a previous version.

# Stop the current version
docker-compose -f docker-compose.prod.yml down

# Restore a previous version (e.g., v0.9)
docker pull registry.company.com/churn-prediction:v0.9
docker tag registry.company.com/churn-prediction:v0.9 churn-prediction:prod
docker-compose -f docker-compose.prod.yml up -d


8. Monitoring Checklist
Use this checklist to verify the health of your service after deployment.

[ ] API health endpoint is responding.

[ ] Prediction latency is < 100ms.

[ ] Error rate is < 1%.

[ ] Model metrics are stable.

[ ] No data drift alerts.

[ ] Disk space is > 20%.

[ ] Memory usage is < 80%.

9. Support
Contact the support team for any issues or questions.

On-call: ml-team@company.com

Slack: #ml-production-support

Documentation: https://wiki.company.com/ml/churn-model