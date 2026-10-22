from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import shutil

from ..db.models import create_job
from ..storage import save_upload
from ..workers.tasks import process_file_async

router = APIRouter(tags=["upload"])

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    
    content_type = file.content_type or ""
    is_image = content_type.startswith("image/")
    is_video = content_type.startswith("video/")

    if not (is_image or is_video):
        raise HTTPException(status_code=400, detail="Only image/* or video/* are supported.")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(file.file, tmp)

    dest = save_upload(tmp_path, file.filename)
    file_type = "video" if is_video else "image"

    job_id = await create_job(filename=dest.name, file_type=file_type, status="queued")

    background_tasks.add_task(process_file_async, str(job_id), str(dest), file_type)

    return {"job_id": str(job_id)}