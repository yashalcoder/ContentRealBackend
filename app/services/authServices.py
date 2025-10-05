import psycopg2
from fastapi import HTTPException,status
from app.models.schemas import UserSignup,LoginRequest
from app.database import get_db_connection
from app.utils.passwHashing import hash_password,verify_password
from app.utils.JWT import create_access_token
from fastapi.responses import JSONResponse

# app/routes/auth.py
from fastapi import APIRouter, Request, HTTPException, status
from app.database import get_db_connection
import psycopg2

router = APIRouter()


def logout(request: Request):
    # Get token from header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    token = auth_header.split(" ")[1]

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Store token in blacklist
        cursor.execute(
            'INSERT INTO "BlacklistedTokens"(token) VALUES (%s) ON CONFLICT DO NOTHING;',
            (token,)
        )
        conn.commit()
        return {"message": "Logout successful"}
    except psycopg2.Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Database error during logout")
    finally:
        cursor.close()
        conn.close()

def signup_user(user:UserSignup):
    conn=get_db_connection()
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        print("✅ Database connection successful")
        hashed_pw = hash_password(user.password)
        cursor.execute(
        '''INSERT INTO "User"("FirstName", "LastName", "Profession", "Email", "Password")
        VALUES (%s, %s, %s, %s, %s)
        RETURNING "id+";
        ''',
        (user.first_name, user.last_name, user.profession, user.email, hashed_pw)
        )

        user_id=cursor.fetchone()[0]
        conn.commit()
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"message": "User registered successfully", "user_id": user_id}
        )


    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
        detail="Email already exists."
    )
    except ValueError:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid data format."
        )
    except Exception:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error."
        )

    finally:
        cursor.close()
        conn.close()


def login_user(data:LoginRequest):
    conn=get_db_connection()
    try:
        cursor=conn.cursor()
        cursor.execute(
                'SELECT "id+","Email","Password" FROM "User" WHERE "Email" = %s',
                (data.email,)
            )
        user_record=cursor.fetchone()
        if not user_record:
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
        user_id, user_email, stored_password = user_record

        if not verify_password(data.password, stored_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")

            # Create JWT token payload
        token_data = {
                "id": user_id,          # ← This is critical!
                "user_id": user_id,
                "email": user_email
         }

        access_token = create_access_token(token_data)

        return {
                "message": "Login successful",
                "access_token": access_token,
                "token_type": "bearer"
            }    
    finally:
            cursor.close()