from pydantic import BaseModel, EmailStr
from typing import Optional
class UserSignup(BaseModel):
    first_name:str
    last_name:str
    profession:str
    email:EmailStr
    password:str


class LoginRequest(BaseModel):
    email: str
    password: str



class ContentCreate(BaseModel):
    FileType: str
    url: Optional[str] = None
    PostContent: str
    UserId: int
    title: str