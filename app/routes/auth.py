# app/routes/auth.py
from fastapi import APIRouter, Request, HTTPException, status
from app.models.schemas import UserSignup, LoginRequest
from app.services.authServices import signup_user, login_user
from app.database import get_db_connection
import psycopg2

router = APIRouter()

@router.post("/signup")
def signup(user: UserSignup):
    return signup_user(user)

@router.post("/login")
def login(data: LoginRequest):
    return login_user(data)

@router.post("/logout")
def logoutfunc(request: Request):
    """Invalidate the JWT token by adding it to the blacklist table"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid token",
        )

    token = auth_header.split(" ")[1]

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO "BlacklistedTokens"(token) VALUES (%s)
               ON CONFLICT DO NOTHING;''',
            (token,),
        )
        conn.commit()
        return {"message": "Logout successful"}
    except psycopg2.Error:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error during logout",
        )
    finally:
        cursor.close()
        conn.close()
