# Churn Prediction Model - Deployment Guide

## Prerequisites

- Docker and Docker Compose installed
- Access to production server
- SSL certificate for HTTPS
- Monitoring infrastructure (Prometheus/Grafana)

## Deployment Steps

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Muhammadali1029/ml-engineering-training.git
cd ml-engineering-training
cd churn-prediction

# Create production environment file
cp .env.example .env.production
# Edit .env.production with production values


# Build and Test
# Build Docker image
docker build -t churn-prediction:prod .

# Run tests
# docker run --rm churn-prediction:prod pytest tests/
docker run --rm -e PYTHONPATH=/app churn-prediction:prod pytest tests/

# Test locally
docker-compose up -d
curl http://localhost:5000/health


# Deploy to Production
# Tag image
docker tag churn-prediction:prod registry.company.com/churn-prediction:v1.0

# Push to registry
docker push registry.company.com/churn-prediction:v1.0

# Deploy on production server
ssh production-server
docker pull registry.company.com/churn-prediction:v1.0
docker-compose -f docker-compose.prod.yml up -d


# Configure Monitoring
# Set up Prometheus scraping:
- job_name: 'churn-prediction'
  static_configs:
    - targets: ['churn-api:5000']

Import Grafana dashboard from monitoring/grafana-dashboard.json

Set up alerts in monitoring/alerts.yml


Setup SSL/TLS
Use nginx as reverse proxy:

server {
    listen 443 ssl;
    server_name api.churnprediction.company.com;
    
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}


Rollback Procedure
If issues occur:

# Stop current version
docker-compose down

# Restore previous version
docker pull registry.company.com/churn-prediction:v0.9
docker tag registry.company.com/churn-prediction:v0.9 churn-prediction:prod
docker-compose up -d


Monitoring Checklist

 API health endpoint responding
 Prediction latency < 100ms
 Error rate < 1%
 Model metrics stable
 No data drift alerts
 Disk space > 20%
 Memory usage < 80%

Support

On-call: ml-team@company.com
Slack: #ml-production-support
Documentation: https://wiki.company.com/ml/churn-model


### Step 5: Run the Monitoring Dashboard

```bash
# Install streamlit if you haven't
pip install streamlit

# Run the dashboard
streamlit run src/monitoring_dashboard.py
Your Final Tasks:

Create all the files above
Run the tests:
bashpytest tests/ -v

Start the monitoring dashboard:
bashstreamlit run src/monitoring_dashboard.py

Create a docker image:
bashdocker build -t churn-prediction:latest .
