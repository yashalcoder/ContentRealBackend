from fastapi import APIRouter, UploadFile, File, Form
from app.services.upload_service import handle_file_upload
from app.utils.JWT import get_current_user
from fastapi import Depends
router = APIRouter()

@router.post("/upload")
async def upload(file: UploadFile = File(None), fileType: str = Form(...), youtubeUrl: str = Form(None),current_user: dict = Depends(get_current_user),no_of_posts: int = Form(1) ):
    try:
        posts_count = int(no_of_posts)
    except ValueError:
        posts_count = 1
    
    print(f"Received no_of_posts: {posts_count}")
    return await handle_file_upload(file, fileType, youtubeUrl, current_user, posts_count)


