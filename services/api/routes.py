from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

from dataset_analyzer.profile.models import DatasetProfile
from planner.models.training_spec import TrainingSpec

from .orchestrator import (
    analyze_dataframe,
    compile_spec,
    plan_pipeline,
    validate_spec,
)

router = APIRouter(prefix="/api")


class PlanRequest(BaseModel):
    dataset_profile: dict[str, Any]
    user_requirement: str = Field(min_length=1)


class ValidateRequest(BaseModel):
    spec: dict[str, Any]


class CompileRequest(BaseModel):
    spec: dict[str, Any]
    user_approved: bool = False


def _parse_spec(raw_spec: dict[str, Any]) -> TrainingSpec:
    try:
        return TrainingSpec.model_validate(raw_spec)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False),
        ) from exc


@router.get("/health")
def health():
    return {"status": "ok", "service": "aiforge-api"}


@router.post("/analyze")
async def analyze_csv(
    file: UploadFile = File(...),
    target: str | None = Form(default=None),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    try:
        contents = await file.read()
        from io import BytesIO

        dataframe = pd.read_csv(BytesIO(contents))

        if dataframe.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV contains no data rows.",
            )

        profile = analyze_dataframe(dataframe, target=target)
        return profile.model_dump(mode="json")

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not analyze CSV: {exc}",
        ) from exc


@router.post("/pipeline/plan")
def create_plan(request: PlanRequest):
    try:
        profile = DatasetProfile.model_validate(
            request.dataset_profile
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False),
        ) from exc

    try:
        result = plan_pipeline(
            profile=profile,
            user_requirement=request.user_requirement,
        )
        return result.model_dump(mode="json")

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Planning failed: {exc}",
        ) from exc


@router.post("/pipeline/validate")
def validate_plan(request: ValidateRequest):
    spec = _parse_spec(request.spec)

    try:
        result = validate_spec(spec)
        return result.model_dump(mode="json")

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Compliance validation failed: {exc}",
        ) from exc


@router.post("/pipeline/compile")
def compile_plan(request: CompileRequest):
    spec = _parse_spec(request.spec)

    try:
        return compile_spec(
            spec=spec,
            user_approved=request.user_approved,
        )

    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Compilation failed: {exc}",
        ) from exc