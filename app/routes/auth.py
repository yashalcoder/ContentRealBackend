from fastapi import APIRouter
from app.models.schemas import UserSignup,LoginRequest
from app.services.authServices import signup_user,login_user

router = APIRouter()

@router.post("/signup")
def signup(user: UserSignup):
    return signup_user(user)

@router.post("/login")
def login(data:LoginRequest):
    return login_user(data)