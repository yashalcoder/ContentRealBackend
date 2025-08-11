from fastapi import APIRouter, UploadFile, File, Form
from app.services.upload_service import handle_file_upload
from app.utils.JWT import get_current_user
from fastapi import Depends
router = APIRouter()

@router.post("/upload")
async def upload(file: UploadFile = File(None), fileType: str = Form(...), youtubeUrl: str = Form(None),current_user: dict = Depends(get_current_user) ):
    return await handle_file_upload(file, fileType, youtubeUrl,current_user)

