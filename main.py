from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.routes.upload import router  
from app.routes.auth import router as auth_router
from app.routes.posts import router as post_router
from app.routes.schedulePosts import router as schedule_router
from app.services.background_jobs import start_background_jobs
from app.routes.ai_bots import router as ai_bots_router
load_dotenv()

app = FastAPI()
print("cahneg in github")
# CORS (frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api")  # Use 'router'
app.include_router(auth_router,prefix="/auth", tags=["Auth"])
app.include_router(post_router)
app.include_router(schedule_router)
app.include_router(ai_bots_router, prefix="/api/ai", tags=["AI Bots"])
@app.on_event("startup")
async def startup_event():
    """Start background jobs when the application starts"""
    print("Starting background jobs...")
    start_background_jobs()
    print("Background jobs started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up when application shuts down"""
    print("Application shutting down...")

@app.get("/")
def root():
    return {"msg": "Backend running successfully"}