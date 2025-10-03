# app/routes/clips_route.py - Filesystem scanning version
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import FileResponse
import os
import json
import time
import re
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from app.utils.JWT import get_current_user
from app.database import get_db_connection

router = APIRouter()

# # Constants - using your actual directory structure
# CLIP_DIR = r"C:\Users\Yashal Rafique\Desktop\BackendContentRealM\clips"
# THUMBNAILS_DIR = os.path.join(CLIP_DIR, "thumbnails")
# CAPTIONS_DIR = os.path.join(CLIP_DIR, "captions")
CLIP_DIR = os.getenv("CLIPS_DIR", "/tmp/clips")
THUMBNAILS_DIR = os.path.join(CLIP_DIR, "thumbnails")
CAPTIONS_DIR = os.path.join(CLIP_DIR, "captions")
print(f"📁 Using clips directory: {CLIP_DIR}")
print(f"📁 Thumbnails directory: {THUMBNAILS_DIR}")
print(f"📁 Captions directory: {CAPTIONS_DIR}")

def validate_filename(filename: str) -> bool:
    """Validate filename for security"""
    return bool(re.match(r'^[a-zA-Z0-9._-]+\.(mp4|vtt|jpg|jpeg|png)$', filename))

def verify_user_owns_content(content_id: int, user_id: int) -> bool:
    """Verify user owns the content"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id FROM "Content" WHERE id = %s AND "UserId" = %s
        ''', (content_id, user_id))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        owns_content = result is not None
        print(f"🔐 User {user_id} owns content {content_id}: {owns_content}")
        return owns_content
    except Exception as e:
        print(f"❌ Error verifying content ownership: {e}")
        return False

def scan_clips_from_filesystem() -> Dict[str, Any]:
    """Scan filesystem for all clips and organize them"""
    print(f"🔍 Scanning filesystem for clips in: {CLIP_DIR}")
    
    all_clips = []
    
    if not os.path.exists(CLIP_DIR):
        print(f"❌ Clips directory doesn't exist: {CLIP_DIR}")
        return {"clips": []}
    
    # Get all MP4 files in the clips directory
    mp4_files = []
    try:
        for filename in os.listdir(CLIP_DIR):
            if filename.lower().endswith('.mp4') and os.path.isfile(os.path.join(CLIP_DIR, filename)):
                mp4_files.append(filename)
        
        print(f"📹 Found {len(mp4_files)} MP4 files")
    except Exception as e:
        print(f"❌ Error listing clips directory: {e}")
        return {"clips": []}
    
    for filename in mp4_files:
        try:
            clip_id = os.path.splitext(filename)[0]  # Remove .mp4 extension
            clip_path = os.path.join(CLIP_DIR, filename)
            thumbnail_path = os.path.join(THUMBNAILS_DIR, f"{clip_id}.jpg")
            caption_path = os.path.join(CAPTIONS_DIR, f"{clip_id}.vtt")
            
            # Get file info
            file_size = os.path.getsize(clip_path) if os.path.exists(clip_path) else 0
            created_time = os.path.getctime(clip_path) if os.path.exists(clip_path) else time.time()
            
            # Check if associated files exist
            thumbnail_exists = os.path.exists(thumbnail_path)
            captions_exists = os.path.exists(caption_path)
            
            clip_data = {
                "id": clip_id,
                "clip": clip_path,
                "thumbnail": thumbnail_path if thumbnail_exists else None,
                "captions": caption_path if captions_exists else None,
                "title": f"Clip {clip_id[:8]}...",
                "reason": "Generated clip",
                "file_size": file_size,
                "filename": filename,
                "exists": True,
                "thumbnail_exists": thumbnail_exists,
                "captions_exists": captions_exists,
                "created_at": datetime.fromtimestamp(created_time).isoformat(),
                "clip_filename": filename,
                "thumbnail_filename": f"{clip_id}.jpg" if thumbnail_exists else None,
                "captions_filename": f"{clip_id}.vtt" if captions_exists else None,
                "duration": 0,  # We don't have duration info from filesystem
                "start": "00:00:00",
                "end": "00:00:00"
            }
            
            all_clips.append(clip_data)
            print(f"📎 Added clip: {filename} ({file_size} bytes)")
            
        except Exception as e:
            print(f"❌ Error processing clip {filename}: {e}")
            continue
    
    print(f"✅ Scanned {len(all_clips)} clips from filesystem")
    return {"clips": all_clips}

@router.get("/content/{content_id}/clips")
async def get_clips_for_content(
    content_id: int, 
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get all clips for a specific content item - now scans filesystem"""
    try:
        print(f"📱 Getting clips for content {content_id}, user {current_user['id']}")
        
        # Verify user owns this content
        if not verify_user_owns_content(content_id, current_user["id"]):
            print(f"❌ Access denied for content {content_id}")
            raise HTTPException(status_code=404, detail="Content not found or access denied")
        
        # Scan filesystem for clips
        clips_data = scan_clips_from_filesystem()
        all_clips = clips_data.get("clips", [])
        clips = [c for c in all_clips if str(c["filename"]).startswith(f"{content_id}_")]

        print(f"🎬 Found {len(clips)} clips from filesystem scan")
        
        # For now, return all clips for any content the user owns
        # In a production system, you'd want to associate clips with specific content
        result = {
            "clips": clips, 
            "total": len(clips),
            "content_id": content_id,
            "source": "filesystem_scan"
        }
        
        print(f"🎯 Returning {len(clips)} clips to frontend")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting clips: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.get("/debug/filesystem")
async def debug_filesystem(current_user: dict = Depends(get_current_user)):
    """Debug endpoint to see what's on the filesystem"""
    try:
        debug_info = {
            "clip_directory": CLIP_DIR,
            "clip_directory_exists": os.path.exists(CLIP_DIR),
            "thumbnails_directory_exists": os.path.exists(THUMBNAILS_DIR),
            "captions_directory_exists": os.path.exists(CAPTIONS_DIR),
            "user_id": current_user["id"]
        }
        
        # List files in each directory
        if os.path.exists(CLIP_DIR):
            debug_info["files"] = {
                "clips": [],
                "thumbnails": [],
                "captions": []
            }
            
            # Main clips directory
            for filename in os.listdir(CLIP_DIR):
                if os.path.isfile(os.path.join(CLIP_DIR, filename)):
                    file_path = os.path.join(CLIP_DIR, filename)
                    debug_info["files"]["clips"].append({
                        "filename": filename,
                        "size": os.path.getsize(file_path),
                        "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    })
            
            # Thumbnails directory
            if os.path.exists(THUMBNAILS_DIR):
                for filename in os.listdir(THUMBNAILS_DIR):
                    if os.path.isfile(os.path.join(THUMBNAILS_DIR, filename)):
                        file_path = os.path.join(THUMBNAILS_DIR, filename)
                        debug_info["files"]["thumbnails"].append({
                            "filename": filename,
                            "size": os.path.getsize(file_path)
                        })
            
            # Captions directory
            if os.path.exists(CAPTIONS_DIR):
                for filename in os.listdir(CAPTIONS_DIR):
                    if os.path.isfile(os.path.join(CAPTIONS_DIR, filename)):
                        file_path = os.path.join(CAPTIONS_DIR, filename)
                        debug_info["files"]["captions"].append({
                            "filename": filename,
                            "size": os.path.getsize(file_path)
                        })
            
            debug_info["file_counts"] = {
                "clips": len(debug_info["files"]["clips"]),
                "thumbnails": len(debug_info["files"]["thumbnails"]),
                "captions": len(debug_info["files"]["captions"])
            }
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e)}

@router.get("/download/{filename}")
async def download_clip(
    filename: str, 
    current_user: dict = Depends(get_current_user)
):
    """Download a clip file with security validation"""
    try:
        print(f"📥 Download request for: {filename}")
        
        if not validate_filename(filename):
            raise HTTPException(status_code=400, detail="Invalid filename format")
        
        # Find the file in various directories
        possible_paths = [
            os.path.join(CLIP_DIR, filename),
            os.path.join(THUMBNAILS_DIR, filename),
            os.path.join(CAPTIONS_DIR, filename)
        ]
        
        actual_path = None
        for path in possible_paths:
            if os.path.exists(path):
                actual_path = path
                print(f"📁 Found file at: {actual_path}")
                break
        
        if not actual_path:
            print(f"❌ File not found in any directory: {filename}")
            print(f"   Searched paths: {possible_paths}")
            raise HTTPException(status_code=404, detail="File not found")
        
        # Basic security: just check if user is authenticated
        # In production, you'd want more sophisticated access control
        
        # Determine media type
        media_type = 'application/octet-stream'
        if filename.lower().endswith('.mp4'):
            media_type = 'video/mp4'
        elif filename.lower().endswith('.vtt'):
            media_type = 'text/vtt'
        elif filename.lower().endswith(('.jpg', '.jpeg')):
            media_type = 'image/jpeg'
        elif filename.lower().endswith('.png'):
            media_type = 'image/png'
        
        print(f"📤 Serving file: {actual_path} as {media_type}")
        
        return FileResponse(
            path=actual_path,
            filename=filename,
            media_type=media_type,
            headers={
                "Cache-Control": "private, max-age=3600",
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
@router.get("/serve/{filename}")
async def serve_clip(filename: str):

    """Serve a clip file for viewing/streaming (not download) - Fixed for VTT"""
    try:
        if not validate_filename(filename):
            raise HTTPException(status_code=400, detail="Invalid filename format")
        
        # Find the file
        possible_paths = [
            os.path.join(CLIP_DIR, filename),
            os.path.join(THUMBNAILS_DIR, filename),
            os.path.join(CAPTIONS_DIR, filename)
        ]
        
        actual_path = None
        for path in possible_paths:
            if os.path.exists(path):
                actual_path = path
                print(f"📁 Serving file from: {actual_path}")
                break
        
        if not actual_path:
            print(f"❌ File not found: {filename}")
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine media type - FIXED VTT MIME TYPE
        media_type = 'application/octet-stream'
        headers = {}
        
        if filename.lower().endswith('.mp4'):
            media_type = 'video/mp4'
            headers["Accept-Ranges"] = "bytes"  # Enable video seeking
        elif filename.lower().endswith('.vtt'):
            media_type = 'text/vtt'  # Correct MIME type for VTT
            headers["Content-Type"] = 'text/vtt; charset=utf-8'
        elif filename.lower().endswith(('.jpg', '.jpeg')):
            media_type = 'image/jpeg'
        elif filename.lower().endswith('.png'):
            media_type = 'image/png'
        
        # Add CORS headers for cross-origin requests
        headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Cache-Control": "public, max-age=3600",
        })
        
        print(f"📤 Serving {filename} as {media_type}")
        
        return FileResponse(
            path=actual_path,
            media_type=media_type,
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error serving file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# # Also add an OPTIONS handler for CORS preflight requests
# @router.options("/serve/{filename}")
# async def serve_clip_options(filename: str):
#     """Handle CORS preflight requests"""
#     return Response(
#         headers={
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Methods": "GET, OPTIONS",
#             "Access-Control-Allow-Headers": "Content-Type",
#         }
#     )
@router.get("/health")
async def clips_health_check():
    """Health check for clips functionality"""
    try:
        # Check directories
        clips_dir_exists = os.path.exists(CLIP_DIR)
        clips_dir_writable = os.access(CLIP_DIR, os.W_OK) if clips_dir_exists else False
        
        # Count files
        clips_count = 0
        total_size = 0
        
        if clips_dir_exists:
            try:
                for root, dirs, files in os.walk(CLIP_DIR):
                    for filename in files:
                        if filename.lower().endswith(('.mp4', '.vtt', '.jpg', '.jpeg', '.png')):
                            file_path = os.path.join(root, filename)
                            if os.path.exists(file_path):
                                clips_count += 1
                                total_size += os.path.getsize(file_path)
            except Exception as e:
                return {
                    "status": "error", 
                    "error": f"Cannot read clips directory: {str(e)}"
                }
        
        return {
            "status": "healthy",
            "clips_directory": {
                "exists": clips_dir_exists,
                "writable": clips_dir_writable,
                "path": os.path.abspath(CLIP_DIR)
            },
            "files": {
                "total_files": clips_count,
                "total_size": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 2)
            },
            "directories": {
                "clips": CLIP_DIR,
                "thumbnails": THUMBNAILS_DIR, 
                "captions": CAPTIONS_DIR
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }