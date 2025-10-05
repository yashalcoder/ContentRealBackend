import jwt
import datetime
from dotenv import load_dotenv
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.database import get_db_connection
import datetime, os
from dotenv import load_dotenv



load_dotenv()  # load SECRET_KEY from .env

SECRET_KEY = os.getenv("SECRET_KEY", "your_default_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Use HTTPBearer instead of OAuth2PasswordBearer
security = HTTPBearer()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
def verify_access_token(token: str):
    """Checks blacklist and decodes token"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Check if token is blacklisted
        cursor.execute('SELECT 1 FROM "BlacklistedTokens" WHERE token = %s', (token,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been blacklisted",
            )

        # Decode and return payload
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    finally:
        cursor.close()
        conn.close()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    
    try:
        payload = verify_access_token(token) 
        user_id = payload.get("id")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid token - missing user ID"
            )
            
        return {"id": user_id, "email": payload.get("email")}
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Token expired"
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail=f"Token invalid: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail=f"Authentication failed: {str(e)}"
        )