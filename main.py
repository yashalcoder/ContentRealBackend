from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os
import time
import re

# Import your routers
from app.routes.upload import router  
from app.routes.auth import router as auth_router
from app.routes.posts import router as post_router
from app.routes.schedulePosts import router as schedule_router
from app.services.background_jobs import start_background_jobs
from app.routes.ai_bots import router as ai_bots_router
from app.routes.clips_route import router as clips_router

load_dotenv()

# Define CLIP_DIR constant
CLIP_DIR = os.path.abspath("clips")
THUMBNAILS_DIR = os.path.join(CLIP_DIR, "thumbnails")
CAPTIONS_DIR = os.path.join(CLIP_DIR, "captions")

# Create directories
os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(THUMBNAILS_DIR, exist_ok=True)
os.makedirs(CAPTIONS_DIR, exist_ok=True)

# Get public base URL
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")

app = FastAPI(title="Content Creator API", version="1.0.0")

print("🚀 Starting Content Creator API...")
print(f"📁 Clips directory: {CLIP_DIR}")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_filename(filename: str) -> bool:
    """Validate filename for security"""
    return bool(re.match(r'^[a-zA-Z0-9._-]+\.(mp4|srt|jpg|jpeg|png)$', filename))

# Static file serving for clips - IMPORTANT: This handles direct access to clip files
try:
    app.mount("/clips", StaticFiles(directory=CLIP_DIR), name="clips")
    print(f"✅ Clips static files mounted at /clips -> {CLIP_DIR}")
except Exception as e:
    print(f"⚠️ Warning: Could not mount clips directory: {e}")

# Additional static routes for subdirectories
try:
    app.mount("/clips/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")
    app.mount("/clips/captions", StaticFiles(directory=CAPTIONS_DIR), name="captions")
    print("✅ Thumbnail and caption directories mounted")
except Exception as e:
    print(f"⚠️ Warning: Could not mount subdirectories: {e}")

# Include routers with proper prefixes
app.include_router(router, prefix="/api", tags=["Upload"])
app.include_router(auth_router, prefix="/auth", tags=["Auth"]) 
app.include_router(post_router, tags=["Posts"])
app.include_router(schedule_router, tags=["Schedule"])
app.include_router(ai_bots_router, prefix="/api/ai", tags=["AI Bots"])

# IMPORTANT: Mount clips router with /api prefix to match frontend expectations
app.include_router(clips_router, prefix="/api/clips", tags=["Clips"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Content Creator API", 
        "version": "1.0.0",
        "status": "running",
        "clips_directory": CLIP_DIR,
        "endpoints": {
            "upload": "/api/upload",
            "clips": "/api/clips",
            "static_clips": "/clips",
            "health": "/api/clips/health"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "clips_directory": {
            "exists": os.path.exists(CLIP_DIR),
            "writable": os.access(CLIP_DIR, os.W_OK) if os.path.exists(CLIP_DIR) else False
        }
    }

# Direct clip access endpoint (alternative to static files)
@app.get("/api/clips/file/{filename}")
async def get_clip_file(filename: str):
    """Direct access to clip files"""
    try:
        if not validate_filename(filename):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Check all possible locations
        possible_paths = [
            os.path.join(CLIP_DIR, filename),
            os.path.join(THUMBNAILS_DIR, filename),
            os.path.join(CAPTIONS_DIR, filename)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # Determine media type
                media_type = 'application/octet-stream'
                if filename.lower().endswith('.mp4'):
                    media_type = 'video/mp4'
                elif filename.lower().endswith('.srt'):
                    media_type = 'text/plain'
                elif filename.lower().endswith(('.jpg', '.jpeg')):
                    media_type = 'image/jpeg'
                elif filename.lower().endswith('.png'):
                    media_type = 'image/png'
                
                return FileResponse(
                    path=path,
                    media_type=media_type,
                    headers={"Cache-Control": "public, max-age=3600"}
                )
        
        raise HTTPException(status_code=404, detail="File not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Debug endpoint for development
@app.get("/api/debug/clips")
async def debug_clips():
    """Debug endpoint to check clips storage"""
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=404, detail="Not found")
    
    # Import here to avoid circular imports
    from app.services.upload_handler import CLIPS_STORAGE
    
    clip_files = []
    if os.path.exists(CLIP_DIR):
        for root, dirs, files in os.walk(CLIP_DIR):
            for file in files:
                if file.lower().endswith(('.mp4', '.srt', '.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    clip_files.append({
                        "filename": file,
                        "path": full_path,
                        "size": os.path.getsize(full_path),
                        "relative_path": os.path.relpath(full_path, CLIP_DIR)
                    })
    
    return {
        "clips_directory": CLIP_DIR,
        "memory_storage": {
            "content_items": len(CLIPS_STORAGE),
            "total_clips": sum(len(data.get("clips", [])) for data in CLIPS_STORAGE.values())
        },
        "filesystem": {
            "total_files": len(clip_files),
            "files": clip_files[:10]  # Show first 10 files
        }
    }

# Start background jobs
@app.on_event("startup")
async def startup_event():
    print("🔄 Starting background jobs...")
    # Commented out to avoid potential issues on Replit
    start_background_jobs()
    print("✅ Application startup complete")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

   