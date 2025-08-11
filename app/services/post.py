from fastapi import APIRouter,Request
from app.database import get_db_connection

router = APIRouter()

def get_user_posts(user_id: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT id, "FileType", "url", "PostContent", "title"
               FROM "Content"
               WHERE "UserId" = %s
               ORDER BY id DESC''',
            (user_id,)
        )
        rows = cursor.fetchall()
        posts = [
            {
                "id": r[0],
                "FileType": r[1],
                "url": r[2],
                "PostContent": r[3],
                "title": r[4]
            }
            for r in rows
        ]
        return {"status": "success", "posts": posts}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        conn.close()

def get_post(post_id: int):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT id, "FileType", "url", "PostContent", "title", "UserId"
               FROM "Content"
               WHERE id = %s
            ''',
            (post_id,)
        )
        row = cursor.fetchone()
        if not row:
            return {"status": "error", "message": "Post not found"}
        
        post = {
            "id": row[0],
            "FileType": row[1],
            "url": row[2],
            "PostContent": row[3],
            "title": row[4],
            "UserId": row[5]
        }
        return {"status": "success", "post": post}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        conn.close()



async def update_post(post_id: int, request: Request):
    data = await request.json()
    title = data.get("title")
    content = data.get("PostContent")

    if not title or not content:
        return {"status": "error", "message": "Title and content are required"}

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            '''UPDATE "Content"
               SET "title" = %s, "PostContent" = %s
               WHERE id = %s
            ''',
            (title, content, post_id)
        )
        conn.commit()
        return {"status": "success", "message": "Post updated successfully"}
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        conn.close()


import requests
from fastapi import Depends
from app.utils.JWT import get_current_user

from app.database import get_db_connection
from datetime import datetime

def fetch_linkedin_access_token(user_id: int):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT access_token, refresh_token, token_expires_at 
            FROM user_platform_tokens 
            WHERE user_id = %s AND platform = 'linkedin'
            LIMIT 1
        ''', (user_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        access_token, refresh_token, token_expires_at = row
        
        # Optional: Check if token expired, you can refresh here if you implement refresh token logic
        if token_expires_at and token_expires_at < datetime.utcnow():
            # TODO: refresh access token using refresh_token and update DB
            return None  # or updated token

        return access_token

    finally:
        cursor.close()
        conn.close()



async def post_to_linkedin(postId: int, current_user=Depends(get_current_user)):
    user_id = current_user["id"]
    
    # Fetch post content from DB
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT "PostContent" FROM "Content" WHERE id = %s AND "UserId" = %s', (postId, user_id))
        row = cursor.fetchone()
        if not row:
            return {"status": "error", "message": "Post not found or not authorized"}
        post_content = row[0]
    finally:
        cursor.close()
        conn.close()

    # Fetch LinkedIn token for user from your user_platform_tokens table
    # (pseudo code)
    linkedin_token = fetch_linkedin_access_token(user_id)  # Implement this function

    if not linkedin_token:
        return {"status": "error", "message": "LinkedIn account not connected"}

    linkedin_api_url = "https://api.linkedin.com/v2/ugcPosts"
    headers = {
        "Authorization": f"Bearer {linkedin_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"
    }
    payload = {
        "author": f"urn:li:person:{linkedin_user_id}",  # You need to get this during OAuth
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": post_content
                },
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }

    resp = requests.post(linkedin_api_url, headers=headers, json=payload)
    if resp.status_code == 201:
        return {"status": "success", "message": "Posted to LinkedIn"}
    else:
        return {"status": "error", "message": f"LinkedIn API error: {resp.text}"}
