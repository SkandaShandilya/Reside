import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import datetime, timezone
from app.dependencies import get_db, require_admin, require_guard
from app.models.user import User
from app.models.model_version import ModelVersion
from app.schemas.model import ModelVersionResponse, ModelTrainRequest, ModelDeployRequest, ModelListResponse

router = APIRouter(prefix="/api/models", tags=["AI Models"])



training_status = {
    "is_training": False,
    "version": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "progress_percent": 0,
    "stage": "Idle",
    "loss": None,
    "dataset_name": None,
    "started_at": None,
}


@router.get("/training-status")
async def get_training_status(
    current_user: User = Depends(require_guard),
):
    return training_status


# The classes the project actually trains on. Fire and Smoke are one
# application-level class ("Fire/Smoke Detected", see app/constants.py).
# Dropped classes (Falling/Snatching/Kicking) and rule-based detectors
# (Loitering/Running/Crowd Formation, which have no trained model) are
# deliberately not reported here.
TRAINING_CLASSES = {
    "Normal": ("Normal",),
    "Fighting": ("Fighting",),
    "Fire/Smoke": ("Fire", "Smoke", "Fire/Smoke"),
}


@router.get("/training-data")
async def get_training_data(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    """Real training-sample counts for the implemented classes only."""
    from sqlalchemy import func, cast, String
    from app.models.training_sample import TrainingSample

    # Read `source` as plain text, lower-cased: rows written by the dataset
    # ingest scripts store the value ("dataset") rather than the enum name
    # the ORM expects, so going through the Enum type raises LookupError.
    source_text = func.lower(cast(TrainingSample.source, String))
    result = await db.execute(
        select(TrainingSample.activity_label, source_text, func.count(TrainingSample.id))
        .group_by(TrainingSample.activity_label, source_text)
    )
    counts: dict[tuple[str, str], int] = {(lbl, src): n for lbl, src, n in result.all()}

    classes = []
    for name, labels in TRAINING_CLASSES.items():
        orig = sum(counts.get((lbl, "dataset"), 0) for lbl in labels)
        cctv = sum(counts.get((lbl, "verified_cctv"), 0) for lbl in labels)
        classes.append({"name": name, "original": orig, "verified_cctv": cctv, "total": orig + cctv})

    return {
        "total": sum(c["total"] for c in classes),
        "original": sum(c["original"] for c in classes),
        "verified_cctv": sum(c["verified_cctv"] for c in classes),
        "class_count": len(classes),
        "classes": classes,
    }


@router.get("", response_model=ModelListResponse)
async def list_models(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    result = await db.execute(select(ModelVersion).order_by(desc(ModelVersion.created_at)))
    models = result.scalars().all()

    deployed = None
    for m in models:
        if m.is_deployed:
            deployed = ModelVersionResponse.model_validate(m)
            break

    return ModelListResponse(
        models=[ModelVersionResponse.model_validate(m) for m in models],
        deployed_model=deployed,
    )


@router.get("/{model_id}", response_model=ModelVersionResponse)
async def get_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_guard),
):
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelVersionResponse.model_validate(model)


@router.post("/train", response_model=ModelVersionResponse, status_code=202)
async def start_training(
    body: ModelTrainRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Execute PyTorch fine-tuning using human-verified CCTV review samples."""
    import asyncio
    import functools
    from app.models.incident_review import IncidentReview
    from ai_service.training.fine_tune import run_fine_tuning
    import json as _json

    # Get highest existing version and increment major
    result = await db.execute(select(ModelVersion).order_by(desc(ModelVersion.created_at)).limit(1))
    latest = result.scalar_one_or_none()
    if latest:
        parts = latest.version.split(".")
        try:
            new_version = f"{int(parts[0]) + 1}.0.0"
        except (ValueError, IndexError):
            new_version = "3.0.0"
        base_version = latest.version
    else:
        new_version = "1.0.0"
        base_version = "1.0.0"

    # Ensure the generated version does not already exist (loop until unique)
    candidate = new_version
    suffix = 0
    while True:
        check = await db.execute(select(ModelVersion).where(ModelVersion.version == candidate))
        if not check.scalar_one_or_none():
            new_version = candidate
            break
        suffix += 1
        parts = candidate.split(".")
        candidate = f"{parts[0]}.{parts[1]}.{suffix}"

    # Construct dataset label name and sample list based on selected_datasets
    dataset_names_map = {
        "smoke_fire": "Smoke & Fire Detection Dataset (YOLO)",
        "rwf_2000": "RWF-2000 Violence Detection Database",
        "ucf_crime": "UCF Crime Dataset",
        "cctv_action": "CCTV Action Recognition Dataset",
        "verified_cctv": "Human-Verified CCTV Review Queue",
        "sentinel_base": "Sentinel Baseline Synthetic Set",
    }


    chosen_names = [dataset_names_map.get(ds, ds) for ds in (body.selected_datasets or ["verified_cctv"])]
    dataset_title = " + ".join(chosen_names) if chosen_names else "Human-Verified CCTV Dataset"

    # Fetch verified samples from database if "verified_cctv" or "sentinel_base" selected
    verified_samples = []
    if "verified_cctv" in (body.selected_datasets or []) or not body.selected_datasets:
        rev_result = await db.execute(
            select(IncidentReview).where(IncidentReview.verified_label != None)  # noqa: E711
        )
        reviews = rev_result.scalars().all()
        verified_samples = [
            {"video_path": r.video_path or "", "label": r.verified_label}
            for r in reviews if r.verified_label
        ]

    global training_status
    training_status.update({
        "is_training": True,
        "version": new_version,
        "current_epoch": 0,
        "total_epochs": body.epochs or 5,
        "progress_percent": 0,
        "stage": "Initializing fine-tuning pipeline...",
        "loss": None,
        "dataset_name": dataset_title,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    def _on_progress(data: dict):
        global training_status
        training_status.update({
            "current_epoch": data.get("epoch", 0),
            "total_epochs": data.get("total_epochs", body.epochs or 5),
            "progress_percent": data.get("progress_percent", 0),
            "stage": data.get("stage", "Training..."),
            "loss": data.get("loss"),
        })

    try:
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(
            None,
            functools.partial(
                run_fine_tuning,
                verified_samples=verified_samples,
                base_version=base_version,
                new_version=new_version,
                epochs=body.epochs or 5,
                progress_callback=_on_progress,
            ),
        )
    finally:
        training_status["is_training"] = False
        training_status["progress_percent"] = 100
        training_status["stage"] = "Completed"


    # Ensure per_class_metrics and activity_classes are Python dicts/lists (not JSON strings)
    per_class = metrics.get("per_class_metrics") or {}
    if isinstance(per_class, str):
        try:
            per_class = _json.loads(per_class)
        except Exception:
            per_class = {}

    act_classes = metrics.get("activity_classes") or []
    if isinstance(act_classes, str):
        try:
            act_classes = _json.loads(act_classes)
        except Exception:
            act_classes = []

    model = ModelVersion(
        version=new_version,
        name=body.name or f"SuspiciousActivity-v{new_version}",
        architecture=body.architecture or "Time-Distributed CNN",
        dataset_name=dataset_title,
        weights_path=metrics.get("weights_path"),


        accuracy=metrics.get("accuracy"),
        val_accuracy=metrics.get("val_accuracy"),
        f1_score=metrics.get("f1_score"),
        precision_score=metrics.get("precision_score"),
        recall_score=metrics.get("recall_score"),
        per_class_metrics=per_class,
        activity_classes=act_classes,
        training_samples_count=metrics.get("training_samples_count"),
        validation_samples_count=metrics.get("validation_samples_count"),
        epochs=metrics.get("epochs"),
        training_duration_seconds=metrics.get("training_duration_seconds"),
        is_deployed=False,
        is_baseline=False,
        trained_at=datetime.now(timezone.utc),
        notes=body.notes or f"Fine-tuned model trained on {len(verified_samples)} human-verified CCTV samples",
    )
    db.add(model)
    await db.flush()
    await db.refresh(model)
    return ModelVersionResponse.model_validate(model)


@router.post("/{model_id}/deploy", response_model=ModelVersionResponse)
async def deploy_model(
    model_id: int,
    body: ModelDeployRequest = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Undeploy all other models
    all_result = await db.execute(select(ModelVersion).where(ModelVersion.is_deployed == True))
    for m in all_result.scalars().all():
        m.is_deployed = False

    model.is_deployed = True
    model.deployed_at = datetime.now(timezone.utc)
    if body and body.notes:
        model.notes = (model.notes or "") + f"\n[DEPLOYED] {body.notes}"

    await db.flush()
    await db.refresh(model)
    return ModelVersionResponse.model_validate(model)


@router.delete("/{model_id}", status_code=204)
async def delete_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Deletes a trained checkpoint's DB row and, best-effort, its .pt weights
    file on disk. The currently DEPLOYED version can never be deleted
    outright — inference_engine.py's checkpoint selection is DB-authoritative
    (see AIConfig._get_deployed_checkpoint), so deleting a deployed row out
    from under it would silently fall back to auto-selecting whatever else
    is on disk instead of failing loudly. Deploy a different version first
    (or roll back) if you want to remove the one currently in use.
    """
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if model.is_deployed:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the currently deployed model. Deploy a different version first.",
        )

    if model.weights_path:
        backend_dir = Path(__file__).resolve().parent.parent.parent
        candidates = [Path(model.weights_path), backend_dir / model.weights_path]
        for candidate in candidates:
            try:
                if candidate.is_file():
                    candidate.unlink()
                    break
            except OSError:
                pass

    await db.delete(model)


@router.post("/{model_id}/rollback", response_model=ModelVersionResponse)
async def rollback_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Rollback to a previous model version."""
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Undeploy all other models
    all_result = await db.execute(select(ModelVersion).where(ModelVersion.is_deployed == True))
    for m in all_result.scalars().all():
        m.is_deployed = False

    model.is_deployed = True
    model.deployed_at = datetime.now(timezone.utc)
    model.notes = (model.notes or "") + "\n[ROLLBACK] Rolled back to this version"

    await db.flush()
    await db.refresh(model)
    return ModelVersionResponse.model_validate(model)
