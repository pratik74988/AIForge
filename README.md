# AIForge

**AIForge** is an AI-assisted MLOps platform that helps users move from a dataset and a plain-language machine-learning goal to a proposed, validated, and compiled ML pipeline.

The current repository includes a React/Vite frontend, a FastAPI backend, dataset-profiling utilities, a pipeline planner, compliance checks, and reusable scikit-learn templates. The repository is under active development; infrastructure and production deployment components may be incomplete.

## What it does

- **Dataset analysis:** Upload a CSV and generate a structured profile of its columns and data characteristics.
- **AI-assisted planning:** Provide a machine-learning objective in plain language and generate a proposed training specification.
- **Feature resolution:** Infer numerical and categorical feature lists from the dataset profile when they are not explicitly supplied.
- **Compliance validation:** Check the proposed training specification against the project's compliance rules.
- **Approval-gated compilation:** Require explicit user approval and successful compliance checks before compiling a pipeline specification.
- **Reusable ML templates:** Provide reusable preprocessing, classification, regression, clustering, and evaluation components.

## Workflow

```text
CSV dataset
    |
    v
Dataset analysis and profiling
    |
    v
User requirement + dataset profile
    |
    v
AI-assisted pipeline planning
    |
    v
Training specification
    |
    v
Compliance validation
    |
    v
Explicit user approval
    |
    v
Pipeline compilation
```

Compilation currently returns JSON-safe metadata about the selected template, algorithm, preprocessing steps, hyperparameters, and evaluation metrics. This API does not, by itself, demonstrate that a model has been trained or deployed.

## Repository structure

```text
AIForge-main/
├── frontend/                 # React 19 + Vite user interface
├── services/
│   └── api/                  # FastAPI app, routes, and orchestration
├── dataset-analyzer/         # Dataset analysis, profiling, and ingestion
├── planner/                  # Pipeline planning and training-spec models
├── policy/                   # Compliance checks and policy logic
├── compiler/                 # Pipeline compilation logic
├── ml-templates/
│   ├── registry/             # ML template registry
│   └── sklearn/              # Reusable scikit-learn templates
├── knowledge-base/           # Curated ML guidance and references
├── platform/                 # Infrastructure-related components
├── schemas/                  # Shared schemas (in progress)
├── docs/                     # Project documentation
├── .env.example              # Example environment variables
└── README.md
```

> Some folders and infrastructure integrations are placeholders or are still being implemented. The exact contents of the repository should be treated as the source of truth.

## Technology stack

| Area | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, CSS |
| API | FastAPI, Python, Pydantic |
| Data analysis | Pandas, NumPy, Pandera |
| ML templates | scikit-learn, MLflow, Joblib |
| Dataset ingestion | CSV, SQL, MongoDB connectors are present in the dataset-analyzer package |
| Quality and testing | pytest, Ruff, ESLint |

The planner and other modules may use external model providers or services. Configure the environment variables required by your local setup before running the complete workflow.

## Prerequisites

- Python **3.12 or newer** for the Python components
- Node.js and npm compatible with the installed Vite version
- Git
- [`uv`](https://docs.astral.sh/uv/) for Python dependency and project management

## Getting started

### 1. Clone the repository

```bash
git clone <repository-url>
cd AIForge-main
```

Replace `<repository-url>` with the URL of your Git repository.

### 2. Configure environment variables

Copy the example environment file and edit it with the values required by your setup.

**Windows PowerShell**
```powershell
Copy-Item .env.example .env
```

**macOS/Linux**
```bash
cp .env.example .env
```

Do not commit `.env` or real API keys to source control. Only configure credentials for integrations you actually use. Check the planner and service configuration to confirm which variables are required.

### 3. Install Python dependencies

The repository uses multiple Python packages/projects. Install the relevant projects in editable mode from the repository root:

```bash
uv pip install -e ./dataset-analyzer
uv pip install -e ./ml-templates/registry
uv pip install -e ./ml-templates/sklearn
```

The API also imports modules from `services`, `planner`, `policy`, and `compiler`. Check each project's `pyproject.toml` and package layout, and install any additional dependencies or workspace packages required by your branch. If your branch defines a root workspace or service-specific dependency file, follow that configuration instead.

### 4. Start the backend API

From the repository root, make sure the Python environment contains the required API dependencies and that the local package imports resolve, then run:

```bash
uvicorn services.api.main:app --reload
```

The API should be available at:

- API root: `http://127.0.0.1:8000/`
- Health check: `http://127.0.0.1:8000/api/health`
- Interactive API docs: `http://127.0.0.1:8000/docs`

A successful health check returns:

```json
{
  "status": "ok",
  "service": "aiforge-api"
}
```

If startup fails because a module or dependency cannot be found, check the package installation and imports for `planner`, `policy`, and `compiler` before proceeding.

### 5. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite prints the local frontend URL in the terminal, commonly `http://localhost:5173`.

To create a production frontend build:

```bash
npm run build
```

To run the frontend linter:

```bash
npm run lint
```

### 6. Run Python tests

Run the tests relevant to the component you changed. For example:

```bash
uv run --project dataset-analyzer pytest
uv run --project ml-templates/sklearn pytest
uv run --project ml-templates/registry pytest
```

Test commands may need adjustment depending on how the Python packages are installed and how the current branch configures its environment.

## API endpoints

The FastAPI router currently exposes these endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Basic API status message |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/analyze` | Analyze an uploaded CSV; optional `target` form field |
| `POST` | `/api/pipeline/plan` | Generate a proposed pipeline from a dataset profile and user requirement |
| `POST` | `/api/pipeline/validate` | Validate a training specification against compliance rules |
| `POST` | `/api/pipeline/compile` | Compile a specification after explicit approval and compliance checks |

### Example: health check

```bash
curl http://127.0.0.1:8000/api/health
```

### Example: analyze a CSV

```bash
curl -X POST "http://127.0.0.1:8000/api/analyze" \
  -F "file=@your_dataset.csv" \
  -F "target=target_column"
```

Replace `your_dataset.csv` and `target_column` with your own file and target column. The `target` field is optional.

### Pipeline endpoint request shapes

`POST /api/pipeline/plan` expects JSON containing:

```json
{
  "dataset_profile": {},
  "user_requirement": "Predict the target column from the available features"
}
```

The `dataset_profile` must match the `DatasetProfile` schema returned by `/api/analyze`; an empty object is shown above only to illustrate the outer request shape.

`POST /api/pipeline/validate` expects:

```json
{
  "spec": {}
}
```

`POST /api/pipeline/compile` expects:

```json
{
  "spec": {},
  "user_approved": true
}
```

For validation and compilation, `spec` must match the project's `TrainingSpec` schema. Compilation is blocked unless `user_approved` is `true`, and the backend re-runs compliance checks rather than trusting a prior UI result.

## Development notes

- The frontend and backend run as separate development processes.
- The API's CORS configuration currently permits localhost frontend origins on ports `5173` and `4173`.
- CSV uploads are currently checked by filename extension and then parsed with Pandas. Do not expose this endpoint publicly without appropriate upload-size limits and additional input/security controls.
- The planner, policy, and compiler implementations determine the exact behavior of planning, compliance, and compilation.
- Keep documentation and tests updated when adding endpoints, changing request schemas, or introducing integrations.

## Troubleshooting

**The API does not start**
- Confirm Python 3.12+ is active.
- Confirm the required Python packages are installed.
- Run the command from the repository root so the `services.api.main:app` import can resolve.
- Check missing imports from `planner`, `policy`, or `compiler`.

**The frontend cannot reach the API**
- Confirm the backend is running.
- Check the API URL used by the frontend and the browser console.
- Confirm the frontend origin is allowed by the backend CORS configuration.

**Pipeline planning fails**
- Confirm the dataset profile came from `/api/analyze` and matches `DatasetProfile`.
- Check the planner's required API keys and environment configuration.
- Inspect the backend response and terminal logs for details.

**Compilation is blocked**
- Confirm the request sets `user_approved` to `true`.
- Submit a valid `TrainingSpec`.
- Resolve compliance errors and any additional human-approval requirements.

## Project status

**Active development.** 
