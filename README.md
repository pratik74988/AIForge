# AI Forge

AI Forge is an agentic MLOps platform designed to automate and orchestrate the machine learning lifecycle from user intent to production deployment.

## Architecture

User
↓
AI Gateway
↓
Planner
↓
Project Specification
↓
Compiler
↓
Execution Plan
↓
Policy Validation
↓
Argo Workflows
↓
Model Training
↓
MLflow / Model Registry
↓
KServe
↓
Monitoring & Drift Detection
↓
Retraining

## Repository Structure

- `apps/web` - Next.js frontend
- `apps/api` - FastAPI gateway
- `services/planner` - LLM, RAG and tools
- `services/compiler` - IR to execution plan and templates
- `services/policy` - OPA policies and validation
- `services/monitoring` - Drift detection and telemetry
- `ml-templates` - ML training templates
- `platform` - Kubernetes, Argo, KServe, Helm, Argo CD and Terraform
- `schemas` - ProjectSpec and ExecutionPlan schemas
- `knowledge-base` - Curated platform standards
- `tests` - Integration, policy and planner evaluation tests
- `docs` - Project documentation

## Tech Stack

### Frontend
- Next.js
- TypeScript

### Backend
- FastAPI
- Python

### AI
- LLM
- RAG
- Tool Calling

### ML
- Scikit-learn
- XGBoost
- MLflow

### Infrastructure
- Kubernetes
- AWS EKS
- AWS ECR
- Argo Workflows
- Argo CD
- KServe
- Helm
- Terraform

### Policy
- Open Policy Agent (OPA)

## Development

Project setup and development instructions will be added as individual components are implemented.

## Status

🚧 Active Development
