from fastapi import APIRouter, UploadFile, File, Form
from app.services.upload_service import handle_file_upload

router = APIRouter()

@router.post("/upload")
async def upload(file: UploadFile = File(None), fileType: str = Form(...), youtubeUrl: str = Form(None)):
    return await handle_file_upload(file, fileType, youtubeUrl)
