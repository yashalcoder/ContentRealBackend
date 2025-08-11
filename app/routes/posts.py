from fastapi import APIRouter, UploadFile, File, Form
from app.services.post import get_user_posts,get_post,update_post,post_to_linkedin
router = APIRouter()
@router.get("/my-posts/{user_id}")
async def getPosts(user_id:int):
      return get_user_posts(user_id)

@router.get("/posts/{post_id}")
async def get_post_byID(post_id:int):
      return get_post(post_id)

@router.put("/posts/{post_id}")
async def update_post_by_ID(post_id:int):
      return update_post(post_id)


@router.post("/post/linkedin")
async def upload_to_linkdin(post_id:int):
      return post_to_linkedin(post_id)