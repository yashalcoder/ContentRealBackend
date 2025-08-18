from fastapi import APIRouter, Depends
from app.database import get_db_connection
from app.models.schemas import PlatformToken
from datetime import datetime
from typing import List
import requests
import os
import json
AYRSHARE_API_KEY="34308F88-FFBF4D4C-AD086DC0-B6E7333A"
# AYRSHARE_API_KEY = os.getenv("AYRSHARE_API_KEY")
AYRSHARE_API_URL = "https://app.ayrshare.com/api/post"

def get_connected_platforms(user_id: int):
    """Get list of connected platforms for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''SELECT platform FROM "usersconnectedplatforms" WHERE user_id = %s''',
            (user_id,)
        )
        rows = cursor.fetchall()
        return {"status": "success", "platforms": [r[0] for r in rows]}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        conn.close()

def save_platform_token(data: PlatformToken):
    """Save platform token after successful connection"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''INSERT INTO "usersconnectedplatforms" (user_id, platform, profile_key)
               VALUES (%s, %s, %s)
               ON CONFLICT (user_id, platform) DO UPDATE SET profile_key = EXCLUDED.profile_key''',
            (data.user_id, data.platform, data.profile_key)
        )
        conn.commit()
        return {"status": "success"}
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        conn.close()

def get_user_profile_key(user_id: int):
    """Get existing profile key for a user (FIXED VERSION)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''SELECT profile_key FROM "usersconnectedplatforms" 
               WHERE user_id = %s LIMIT 1''',
            (user_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else None
    except Exception as e:
        print(f"Error getting profile key: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def create_ayrshare_profile(user_id: int):
    """Create a new Ayrshare profile for user"""
    try:
        profile_data = {"title": f"User {user_id} Profile"}
        
        response = requests.post(
            "https://app.ayrshare.com/api/profiles/profile",
            headers={
                "Authorization": f"Bearer {AYRSHARE_API_KEY}",
                "Content-Type": "application/json"
            },
            json=profile_data
        )
        
        if response.status_code == 201:
            return response.json().get("profileKey")
        elif response.status_code == 409:
            # Profile already exists, try to get existing one
            existing_key = get_user_profile_key(user_id)
            if existing_key:
                return existing_key
            else:
                raise Exception("Profile exists but no key found in database")
        else:
            raise Exception(f"Failed to create profile: {response.text}")
            
    except Exception as e:
        raise Exception(f"Error creating Ayrshare profile: {str(e)}")
import requests
import json
from typing import List

# Global API key from .env
AYRSHARE_API_KEY = "34308F88-FFBF4D4C-AD086DC0-B6E7333A"
AYRSHARE_API_URL = "https://api.ayrshare.com/api/post"
# publish_to_ayrshare(content: str, media_url: str, profile_keys: list):
#  """Publish content to social media platforms via Ayrshare""" 
# payload = { "post": content, "profileKeys": profile_keys,
#  # Use profileKeys instead of platforms } 
# if media_url:
#  payload["mediaUrls"] = [media_url] headers = { "Authorization": f"Bearer {AYRSHARE_API_KEY}", "Content-Type": "application/json" }
#  try: response = requests.post(AYRSHARE_API_URL, json=payload, headers=headers) if response.status_code == 200: 
# return {"status": "success", "data": response.json()}
#  else: return {"status": "error", "message": f"Ayrshare API error: {response.text}"}
#  except requests.exceptions.RequestException as e: return {"status": "error", "message": f"Request failed:
def publish_to_ayrshare(content: str, media_url: str = None, platforms: List[str] = None, profile_keys: List[str] = None):
    """
    Publish content to social media via Ayrshare.
    
    Logic:
    - If profile_keys provided (paid plan) => use them
    - Else (free plan/testing) => use global API key + platforms
    """
    if profile_keys:
        # Paid plan, use profile keys
        payload = {
            "post": content,
            "profileKeys": profile_keys
        }
        if media_url:
            payload["mediaUrls"] = [media_url]
    else:
        # Free plan / testing, use global API key with platforms
        payload = {
            "post": content,
            "platforms": platforms or ["linkedin"]  # Default to LinkedIn if not specified
        }
        if media_url:
            payload["mediaUrls"] = [media_url]

    headers = {
        "Authorization": f"Bearer {AYRSHARE_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(AYRSHARE_API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        else:
            return {"status": "error", "message": f"Ayrshare API error: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {str(e)}"}


def publish_pending_posts():
    """Background task to publish scheduled posts"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get pending posts that are ready to be published
        cursor.execute(
            '''SELECT id, user_id, content, media_url, platforms
               FROM "scheduledposts"
               WHERE status = 'pending' AND scheduled_at <= NOW()'''
        )
        posts = cursor.fetchall()

        for post in posts:
            post_id, user_id, content, media_url, platforms = post
            
            # Parse platforms JSON
            if isinstance(platforms, str):
                platforms = json.loads(platforms)
            elif platforms is None:
                platforms = []

            # Fetch profile keys for each platform for this user
            cursor.execute(
                '''SELECT profile_key FROM "usersconnectedplatforms"
                   WHERE user_id = %s AND platform = ANY(%s)''',
                (user_id, platforms)
            )
            
            profile_key_rows = cursor.fetchall()
            profile_keys = [r[0] for r in profile_key_rows]

            if not profile_keys:
                print(f"No profile keys found for user {user_id} and platforms {platforms}")
                cursor.execute(
                    '''UPDATE "scheduledposts" 
                       SET status = 'failed', error_message = 'No connected platforms found'
                       WHERE id = %s''',
                    (post_id,)
                )
                conn.commit()
                continue

            # Publish to Ayrshare
            result = publish_to_ayrshare(content, media_url, profile_keys=None)

            if result["status"] == "success":
                post_ids_json = json.dumps(result["data"].get("postIds", {}))
                cursor.execute(
                    '''UPDATE "scheduledposts" 
                       SET status = 'posted', post_ids = %s
                       WHERE id = %s''',
                    (post_ids_json, post_id)
                )
                conn.commit()
                print(f"Successfully posted scheduled post {post_id} for user {user_id}")
            else:
                cursor.execute(
                    '''UPDATE "scheduledposts" 
                       SET status = 'failed', error_message = %s
                       WHERE id = %s''',
                    (result["message"], post_id)
                )
                conn.commit()
                print(f"Error posting scheduled post {post_id}: {result['message']}")

    except Exception as e:
        print("Error in publish_pending_posts:", e)
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

async def ayrshare_callback(request):
    """Handle Ayrshare callback after platform connection"""
    params = dict(request.query_params)
    profile_key = params.get("profileKey")
    platform = params.get("platform")
    user_id = params.get("user_id")

    if not profile_key or not platform or not user_id:
        return {"status": "error", "message": "Missing required parameters"}

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            '''INSERT INTO "usersconnectedplatforms" (user_id, platform, profile_key)
               VALUES (%s, %s, %s)
               ON CONFLICT (user_id, platform)
               DO UPDATE SET profile_key = EXCLUDED.profile_key''',
            (int(user_id), platform, profile_key)
        )
        conn.commit()
        
        return {"status": "success", "message": f"{platform} connected successfully"}
        
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        conn.close()
