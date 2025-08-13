from fastapi import APIRouter, Depends
from app.database import get_db_connection
from app.models.schemas import PlatformToken
from datetime import datetime
from typing import List
import requests
import os
import json


AYRSHARE_API_KEY = os.getenv("AYRSHARE_API_KEY")  # from your Ayrshare dashboard
AYRSHARE_API_URL = "https://app.ayrshare.com/api/post"

def get_connected_platforms(user_id: int):
    """Get list of connected platforms for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''SELECT platform 
               FROM "usersconnectedplatforms" 
               WHERE user_id = %s''',
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


def publish_to_ayrshare(content: str, media_url: str, profile_keys: list):
    """Publish content to social media platforms via Ayrshare"""
    payload = {
        "post": content,
        "platforms": profile_keys,  # These should be the profile keys from connected platforms
    }
    
    # Add media if provided
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
    """Background task to publish scheduled posts - matches your scheduledposts table structure"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Get pending posts that are ready to be published
        cursor.execute(
            '''SELECT id, user_id, content, media_url, platforms
               FROM "scheduledposts"
               WHERE status = 'pending' AND scheduled_at <= NOW()''',
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
                '''SELECT profile_key 
                   FROM "usersconnectedplatforms"
                   WHERE user_id = %s AND platform = ANY(%s)''',
                (user_id, platforms)
            )
            profile_key_rows = cursor.fetchall()
            profile_keys = [r[0] for r in profile_key_rows]

            if not profile_keys:
                print(f"No profile keys found for user {user_id} and platforms {platforms}")
                # Update status to failed
                cursor.execute(
                    '''UPDATE "scheduledposts" 
                       SET status = 'failed'
                       WHERE id = %s''',
                    (post_id,)
                )
                conn.commit()
                continue

            # Publish to Ayrshare
            result = publish_to_ayrshare(content, media_url, profile_keys)

            if result["status"] == "success":
                # Update status to posted
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
                # Update status to failed
                cursor.execute(
                    '''UPDATE "scheduledposts" 
                       SET status = 'failed'
                       WHERE id = %s''',
                    (post_id,)
                )
                conn.commit()
                print(f"Error posting scheduled post {post_id} for user {user_id}: {result['message']}")

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
    user_id = params.get("user_id")  # Pass this in the initial connect URL

    if not profile_key or not platform or not user_id:
        return {"status": "error", "message": "Missing required parameters"}

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Save the platform connection to usersconnectedplatforms table
        cursor.execute(
            '''INSERT INTO "usersconnectedplatforms" (user_id, platform, profile_key)
               VALUES (%s, %s, %s)
               ON CONFLICT (user_id, platform)
               DO UPDATE SET profile_key = EXCLUDED.profile_key''',
            (int(user_id), platform, profile_key)
        )
        conn.commit()
        
        return {
            "status": "success",
            "message": f"{platform} connected successfully"
        }
        
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        conn.close()


def get_user_analytics(user_id: int):
    """Get analytics for a user's posts and schedules"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get connected platforms count
        cursor.execute(
            '''SELECT COUNT(*) FROM "usersconnectedplatforms" WHERE user_id = %s''',
            (user_id,)
        )
        connected_platforms = cursor.fetchone()[0]
        
        # Get scheduled posts stats
        cursor.execute(
            '''SELECT status, COUNT(*) FROM "scheduledposts" 
               WHERE user_id = %s GROUP BY status''',
            (user_id,)
        )
        status_counts = dict(cursor.fetchall())
        
        # Get total content created
        cursor.execute(
            '''SELECT COUNT(*) FROM "Content" WHERE "UserId" = %s''',
            (user_id,)
        )
        total_content = cursor.fetchone()[0]
        
        return {
            "status": "success",
            "analytics": {
                "connected_platforms": connected_platforms,
                "total_content": total_content,
                "scheduled_posts": status_counts.get('pending', 0),
                "published_posts": status_counts.get('posted', 0),
                "failed_posts": status_counts.get('failed', 0)
            }
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        conn.close()


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


def delete_scheduled_post(post_id: int, user_id: int):
    """Delete a scheduled post"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            '''DELETE FROM "scheduledposts" 
               WHERE id = %s AND user_id = %s AND status = 'pending' ''',
            (post_id, user_id)
        )
        conn.commit()
        
        if cursor.rowcount > 0:
            return {"status": "success", "message": "Scheduled post deleted"}
        else:
            return {"status": "error", "message": "Post not found or already published"}
            
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        conn.close()