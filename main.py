from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.routes.upload import router  
from app.routes.auth import router as auth_router
from app.routes.posts import router as post_router
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
@app.get("/")
def root():
    return {"msg": "Backend running successfully"}