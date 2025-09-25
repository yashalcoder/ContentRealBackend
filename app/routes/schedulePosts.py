from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from app.database import get_db_connection
from app.services.schedule import get_connected_platforms, save_platform_token, ayrshare_callback, publish_to_ayrshare,get_user_profile_key,create_ayrshare_profile
from app.models.schemas import (
    PlatformToken, SchedulePostRequest, DisconnectPlatformRequest, 
    InstantPostRequest, ConnectedPlatformsListResponse, APIResponse
)
from datetime import datetime
from typing import List
import os
import requests
import json

router = APIRouter()

AYRSHARE_API_KEY = os.getenv("AYRSHARE_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

@router.get("/getConnectedPlatforms")
async def connected_platforms(user_id: int):
    """Get all connected platforms for a user"""
    return get_connected_platforms(user_id)

@router.post("/savePlatformToken")
async def save_platform_token_route(data: PlatformToken):
    """Save platform token after successful connection"""
    return save_platform_token(data)

@router.get("/ayrshare/callback")
async def ayrshare_callback_route(request: Request):
    """Handle Ayrshare callback after platform connection"""
    return await ayrshare_callback(request)



def get_platform_profile_keys(user_id: int, platforms: List[str]):
    """Get profile keys for specific platforms for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            '''SELECT platform, profile_key FROM "usersconnectedplatforms" 
               WHERE user_id = %s AND platform = ANY(%s)''',
            (user_id, platforms)
        )
        
        results = cursor.fetchall()
        return {platform: profile_key for platform, profile_key in results}
        
    except Exception as e:
        print(f"Error getting profile keys: {e}")
        return {}
    finally:
        cursor.close()
        conn.close()

@router.get("/connect-platform/{platform}")
async def connect_platform(platform: str, user_id: int, request: Request):
    """Generate Ayrshare JWT URL for connecting a platform - FIXED VERSION"""
    try:
        if platform not in ["facebook", "instagram", "linkedin", "twitter"]:
            raise HTTPException(status_code=400, detail="Invalid platform")
        
        base_url = "https://1fbca9e95506.ngrok-free.app"  # Replace with your actual domain
        callback_url = f"{base_url}/ayrshare/callback?user_id={user_id}&platform={platform}"
        
        # Get or create profile key for user
        profile_key = get_user_profile_key(user_id)
        
        if not profile_key:
            # Create new profile
            profile_key = create_ayrshare_profile(user_id)
        
        # Generate JWT URL for social linking
        jwt_data = {
            "domain": base_url,
            "redirectUri": callback_url,
            "platforms": [platform]
        }
        
        jwt_response = requests.post(
            "https://app.ayrshare.com/api/profiles/generateJWT",
            headers={
                "Authorization": f"Bearer {AYRSHARE_API_KEY}",
                "Profile-Key": profile_key,
                "Content-Type": "application/json"
            },
            json=jwt_data
        )
        
        if jwt_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to generate JWT URL")
        
        jwt_url = jwt_response.json().get("url")
        
        return {
            "status": "success",
            "connect_url": jwt_url,
            "profile_key": profile_key
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.delete("/disconnect-platform")
async def disconnect_platform(request: Request):
    """Disconnect a platform for a user"""
    try:
        data = await request.json()
        user_id = data.get("user_id")
        platform = data.get("platform")
        
        if not user_id or not platform:
            raise HTTPException(status_code=400, detail="Missing user_id or platform")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Delete from usersconnectedplatforms table
            cursor.execute(
                '''DELETE FROM "usersconnectedplatforms" 
                   WHERE user_id = %s AND platform = %s''',
                (user_id, platform)
            )
            conn.commit()
            
            if cursor.rowcount > 0:
                return {"status": "success", "message": f"{platform} disconnected successfully"}
            else:
                return {"status": "error", "message": "Platform not found or already disconnected"}
                
        except Exception as e:
            conn.rollback()
            return {"status": "error", "message": str(e)}
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/schedule-post")
async def schedule_post(request: Request):
    """Schedule a post for later publishing"""
    try:
        data = await request.json()
        content_id = data.get("content_id")  # Optional reference to Content table
        user_id = data.get("user_id")
        content = data.get("content")
        media_url = data.get("media_url")
        platforms = data.get("platforms", [])
        scheduled_at = data.get("scheduled_at")
        
        if not all([user_id, content_id, platforms, scheduled_at]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # Parse scheduled datetime
        try:

            scheduled_datetime = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid datetime format")
        
        # Check if scheduled time is in the future
        now_utc = datetime.now(timezone.utc)
        if scheduled_datetime <= now_utc:
            raise HTTPException(status_code=400, detail="Scheduled time must be in the future")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # If content_id is provided, get content from Content table
            if content_id:
                cursor.execute(
                    '''SELECT "PostContent", url FROM "Content" WHERE id = %s AND "UserId" = %s''',
                    (content_id, user_id)
                )
                content_data = cursor.fetchone()
                
                if content_data:
                    content = content_data[0]
                    media_url = media_url or content_data[1]
            
            # Insert into scheduledposts table
            cursor.execute(
                '''INSERT INTO "scheduledposts" (user_id, content, media_url, platforms, scheduled_at, status)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING id''',
                (user_id, content, media_url, json.dumps(platforms), scheduled_datetime, 'pending')
            )
            
            scheduled_post_id = cursor.fetchone()[0]
            conn.commit()
            
            return {
                "status": "success",
                "message": "Post scheduled successfully",
                "scheduled_post_id": scheduled_post_id
            }
            
        except Exception as e:
            conn.rollback()
            return {"status": "error", "message": str(e)}
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/scheduled-posts")
async def get_scheduled_posts(user_id: int):
    """Get all scheduled posts for a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                '''SELECT id, user_id, content, media_url, platforms, scheduled_at, status, post_ids, created_at
                   FROM "scheduledposts" 
                   WHERE user_id = %s
                   ORDER BY scheduled_at DESC''',
                (user_id,)
            )
            
            posts = cursor.fetchall()
            
            scheduled_posts = []
            for post in posts:
                scheduled_posts.append({
                    "id": post[0],
                    "user_id": post[1],
                    "content": post[2],
                    "media_url": post[3],
                    "platforms": post[4] or [],   # Directly use JSONB
                    "scheduled_at": post[5].isoformat() if post[5] else None,
                    "status": post[6],
                    "post_ids": post[7] or None,  # Directly use JSONB
                    "created_at": post[8].isoformat() if post[8] else None
                })

            
            return {
                "status": "success",
                "scheduled_posts": scheduled_posts
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.delete("/scheduled-posts/{scheduled_post_id}")
async def cancel_scheduled_post(scheduled_post_id: int, user_id: int):
    """Cancel a scheduled post"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                '''DELETE FROM "scheduledposts" 
                   WHERE id = %s AND user_id = %s AND status = 'pending' ''',
                (scheduled_post_id, user_id)
            )
            conn.commit()
            
            if cursor.rowcount > 0:
                return {"status": "success", "message": "Scheduled post cancelled successfully"}
            else:
                return {"status": "error", "message": "Scheduled post not found or already posted"}
                
        except Exception as e:
            conn.rollback()
            return {"status": "error", "message": str(e)}
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/publish-now")
async def publish_now(request: Request):
    """Immediately publish a post to selected platforms"""
    try:
        data = await request.json()
        content_id = data.get("content_id")
        user_id = data.get("user_id")
        platforms = data.get("platforms", [])
        content = data.get("content")
        media_url = data.get("media_url")
        
        if not all([user_id, platforms]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # If content_id provided, get content from Content table
            if content_id:
                cursor.execute(
                    '''SELECT "PostContent", url FROM "Content" WHERE id = %s AND "UserId" = %s''',
                    (content_id, user_id)
                )
                content_data = cursor.fetchone()
                
                if not content_data:
                    raise HTTPException(status_code=404, detail="Content not found")
                
                content = content_data[0]
                media_url = media_url or content_data[1]
            
            if not content:
                raise HTTPException(status_code=400, detail="No content provided")
            
            # Get profile keys for selected platforms
            cursor.execute(
                '''SELECT profile_key FROM "usersconnectedplatforms"
                   WHERE user_id = %s AND platform = ANY(%s)''',
                (user_id, platforms)
            )
            
            profile_key_rows = cursor.fetchall()
            profile_keys = [r[0] for r in profile_key_rows]
            
            if not profile_keys:
                return {"status": "error", "message": "No connected platforms found"}
            
            # Publish immediately via Ayrshare
            result = publish_to_ayrshare(content, media_url, profile_keys)
            
            if result["status"] == "success":
                return {
                    "status": "success",
                    "message": "Post published successfully",
                    "data": result["data"]
                }
            else:
                return {"status": "error", "message": result["message"]}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/user-posts")
async def get_user_posts(user_id: int):
    """Get all posts created by a user from Content table"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                '''SELECT id, "FileType", url, "PostContent", title
                   FROM "Content" 
                   WHERE "UserId" = %s
                   ORDER BY id DESC''',
                (user_id,)
            )
            
            posts = cursor.fetchall()
            
            user_posts = []
            for post in posts:
                user_posts.append({
                    "id": post[0],
                    "FileType": post[1],
                    "url": post[2],
                    "PostContent": post[3],
                    "title": post[4]
                })
            
            return {
                "status": "success",
                "posts": user_posts
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return {"status": "error", "message": str(e)}