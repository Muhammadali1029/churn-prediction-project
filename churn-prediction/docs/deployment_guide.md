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