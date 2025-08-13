from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

# User related schemas
class UserSignup(BaseModel):
    first_name: str
    last_name: str
    profession: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    FirstName: str
    LastName: str
    Profession: str
    Email: str
    created_at: Optional[datetime] = None

# Content related schemas
class ContentCreate(BaseModel):
    FileType: str
    url: Optional[str] = None
    PostContent: str
    UserId: int
    title: str

class ContentResponse(BaseModel):
    id: int
    FileType: str
    url: Optional[str] = None
    PostContent: str
    UserId: int
    title: str

class ContentUpdate(BaseModel):
    FileType: Optional[str] = None
    url: Optional[str] = None
    PostContent: Optional[str] = None
    title: Optional[str] = None

# Platform Token schemas
class PlatformToken(BaseModel):
    user_id: int
    platform: str
    profile_key: str

class ConnectedPlatformResponse(BaseModel):
    id: int
    user_id: int
    platform: str
    profile_key: str
    created_at: Optional[datetime] = None

class ConnectedPlatformsListResponse(BaseModel):
    status: str
    platforms: List[str]

class DisconnectPlatformRequest(BaseModel):
    user_id: int
    platform: str

# Scheduled Posts schemas
class SchedulePostRequest(BaseModel):
    content_id: Optional[int] = None  # Reference to Content table
    user_id: int
    content: str
    media_url: Optional[str] = None
    platforms: List[str]  # This will be stored as JSONB
    scheduled_at: datetime

class ScheduledPostResponse(BaseModel):
    id: int
    user_id: int
    content: str
    media_url: Optional[str] = None
    platforms: List[str]  # Convert from JSONB
    scheduled_at: datetime
    status: str
    post_ids: Optional[Dict[str, Any]] = None  # Convert from JSONB
    created_at: Optional[datetime] = None

class ScheduledPostUpdate(BaseModel):
    content: Optional[str] = None
    media_url: Optional[str] = None
    platforms: Optional[List[str]] = None
    scheduled_at: Optional[datetime] = None
    status: Optional[str] = None

class ScheduledPostsListResponse(BaseModel):
    status: str
    scheduled_posts: List[ScheduledPostResponse]

# API Response schemas
class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class AyrshareResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class ConnectPlatformResponse(BaseModel):
    status: str
    connect_url: str

# Instant posting schemas (for immediate posting without scheduling)
class InstantPostRequest(BaseModel):
    content: str
    media_url: Optional[str] = None
    platforms: List[str]
    user_id: int

class PostPublishRequest(BaseModel):
    content_id: int  # Reference to Content table
    platforms: List[str]
    user_id: int
    schedule_now: bool = True  # True for instant, False for scheduling

# Analytics schemas (for future use)
class PlatformAnalytics(BaseModel):
    platform: str
    posts_count: int
    engagement_rate: Optional[float] = None
    reach: Optional[int] = None

class UserAnalyticsResponse(BaseModel):
    status: str
    user_id: int
    total_posts: int
    scheduled_posts: int
    published_posts: int
    failed_posts: int
    platforms: List[PlatformAnalytics]

# Webhook schemas (for Ayrshare callbacks)
class AyrshareWebhookData(BaseModel):
    profileKey: str
    platform: str
    user_id: str
    status: Optional[str] = None
    postId: Optional[str] = None

# Error response schema
class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None