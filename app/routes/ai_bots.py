# app/routes/ai_bots.py
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
import openai
import os
from typing import List, Optional, Dict
from datetime import datetime
import json
from app.database import get_db_connection
from app.utils.JWT import get_current_user
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
print("open ai key in ai", os.getenv("OPENAI_API_KEY"))
router = APIRouter()
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatRequest(BaseModel):
    message: str
    bot_type: str  # coach, drop, search, rep, cast
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    bot_name: str
    suggestions: List[str]
    session_id: str
    user_context_used: bool

class UserAIProfile(BaseModel):
    industry: Optional[str] = None
    specialization: Optional[str] = None
    experience_level: Optional[str] = None
    goals: List[str] = []
    content_types: List[str] = []

def get_user_ai_profile(user_id: int) -> Dict:
    """Get user's AI profile from database"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT industry, specialization, experience_level, goals, content_types
            FROM user_ai_profiles WHERE user_id = %s
        ''', (user_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                "industry": row[0] or "general",
                "specialization": row[1] or "general",
                "experience_level": row[2] or "intermediate",
                "goals": row[3] or [],
                "content_types": row[4] or []
            }
        else:
            # Create default profile if none exists
            return {
                "industry": "general",
                "specialization": "general", 
                "experience_level": "intermediate",
                "goals": [],
                "content_types": []
            }
    finally:
        cursor.close()
        conn.close()

def get_user_content_context(user_id: int) -> str:
    """Get user's recent content for AI context"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Get recent content with analytics if available
        cursor.execute('''
            SELECT c.title, c."FileType", c."PostContent", ca.topics, ca.ai_summary
            FROM public."Content" c
            LEFT JOIN content_analytics ca ON c.id = ca.content_id
            WHERE c."UserId" = %s
            ORDER BY c.id DESC
            LIMIT 5
        ''', (user_id,))
        rows = cursor.fetchall()
        
        if not rows:
            return "No previous content uploaded yet."
        
        context_summary = []
        for row in rows:
            title, file_type, content, topics, summary = row
            topics_str = ", ".join(topics) if topics else "general"
            summary_str = summary if summary else content[:100] + "..."
            
            context_summary.append(f"- {file_type.upper()}: '{title}' (Topics: {topics_str}) - {summary_str}")
        
        return f"User's recent content:\n" + "\n".join(context_summary)
    finally:
        cursor.close()
        conn.close()

def create_personalized_prompt(bot_type: str, user_profile: Dict, user_context: str) -> str:
    """Create personalized AI prompt based on user profile and content"""
    
    industry = user_profile.get('industry', 'general')
    specialization = user_profile.get('specialization', 'general')
    experience = user_profile.get('experience_level', 'intermediate')
    goals = user_profile.get('goals', [])
    
    base_prompts = {
        "coach": f"""You are RealmGPT Coach™, an AI speaking coach specialized for {industry} professionals.

USER PROFILE:
- Industry: {industry.title()}
- Specialization: {specialization.title()}
- Experience Level: {experience.title()}
- Goals: {', '.join(goals) if goals else 'Improve speaking skills'}

USER'S CONTENT HISTORY:
{user_context}

Based on their profile and uploaded content, provide personalized coaching advice. Help them:
- Improve presentation skills for {industry} audiences
- Practice speeches relevant to {specialization}
- Get feedback tailored to their {experience} level
- Achieve their specific goals

Always reference their actual content when giving advice. Be encouraging and specific.""",

        "drop": f"""You are RealmDrop™, an AI content repurposing specialist for {industry} content creators.

USER PROFILE:
- Industry: {industry.title()}
- Specialization: {specialization.title()}
- Content Focus: {specialization}

USER'S CONTENT HISTORY:
{user_context}

Based on their uploaded content, help them:
- Repurpose their {specialization} content into multiple formats
- Create platform-specific posts for {industry} audiences
- Extract key quotes and moments from their talks
- Optimize content for maximum engagement in {industry}

Always reference their actual uploaded content and suggest specific repurposing ideas.""",

        "search": f"""You are RealmSearch™, an AI opportunity finder for {industry} professionals specializing in {specialization}.

USER PROFILE:
- Industry: {industry.title()}
- Specialization: {specialization.title()}
- Experience: {experience.title()}

USER'S CONTENT HISTORY:
{user_context}

Based on their expertise and content topics, help them find:
- Speaking opportunities in {industry}
- {specialization} conferences and events
- Podcasts related to their content themes
- Events suitable for {experience} level speakers

Always suggest opportunities that match their actual content and expertise.""",

        "rep": f"""You are RealmRep™, a professional reputation manager for {industry} professionals.

USER PROFILE:
- Industry: {industry.title()}
- Specialization: {specialization.title()}
- Experience: {experience.title()}

USER'S CONTENT HISTORY:
{user_context}

Help them manage their professional reputation by:
- Creating compelling speaker bios highlighting their {specialization} expertise
- Managing bookings for {industry} events
- Building testimonials from their {industry} work
- Positioning them as a {specialization} expert

Always use their actual content and achievements when building their reputation.""",

        "cast": f"""You are RealmCast™, a streaming and recording assistant for {industry} content creators.

USER PROFILE:
- Industry: {industry.title()}
- Content Focus: {specialization.title()}
- Experience: {experience.title()}

USER'S CONTENT HISTORY:
{user_context}

Based on their content style, help them:
- Set up streaming for {industry} audiences
- Plan recording sessions for {specialization} content
- Choose the right platforms for {industry} reach
- Engage viewers interested in {specialization}

Always tailor technical advice to their content style and {industry} audience."""
    }
    
    return base_prompts.get(bot_type, base_prompts["coach"])
# Initialize the async client
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY") # or use environment variable
)

async def get_ai_response(prompt: str, message: str) -> tuple:
    try:
        # Use the client instance
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",   # or "gpt-4"
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=600,
            temperature=0.7
        )
        return response.choices[0].message.content, True
    except Exception as e:
        print("OpenAI API Error:", e)
        return "I'm having trouble processing your request. Please try again later.", False

def generate_personalized_suggestions(bot_type: str, user_profile: Dict) -> List[str]:
    """Generate personalized suggestions based on user profile"""
    industry = user_profile.get('industry', 'general')
    specialization = user_profile.get('specialization', 'general')
    
    suggestions_map = {
        "coach": [
            f"Help me practice my {specialization} presentation",
            f"Improve my speaking for {industry} audiences",
            f"Structure my {specialization} keynote better",
            "Overcome stage fright and nervousness"
        ],
        "drop": [
            f"Turn my {specialization} content into social clips",
            f"Create {industry} LinkedIn posts from my videos",
            f"Extract key quotes from my {specialization} talks",
            f"Repurpose content for {industry} platforms"
        ],
        "search": [
            f"Find {industry} conferences in 2025",
            f"Podcasts about {specialization}",
            f"Speaking events for {industry} professionals",
            f"{specialization} networking opportunities"
        ],
        "rep": [
            f"Create my {industry} speaker bio",
            f"Handle {specialization} booking inquiries",
            f"Build testimonials for {industry} work",
            f"Position me as {specialization} expert"
        ],
        "cast": [
            f"Stream setup for {industry} content",
            f"Record {specialization} videos better",
            f"Engage {industry} live audiences",
            f"Choose platform for {specialization} content"
        ]
    }
    
    return suggestions_map.get(bot_type, [])

def save_chat_message(session_id: str, user_id: int, bot_type: str, user_message: str, bot_response: str, suggestions: List[str]):
    """Save chat messages to database"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Ensure session exists
        cursor.execute('''
            INSERT INTO ai_chat_sessions (session_id, user_id, bot_type, last_activity)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (session_id) DO UPDATE SET last_activity = NOW()
        ''', (session_id, user_id, bot_type))
        
        # Save user message
        cursor.execute('''
            INSERT INTO ai_chat_messages (session_id, message_type, content, timestamp)
            VALUES (%s, 'user', %s, NOW())
        ''', (session_id, user_message))
        
        # Save bot response
        cursor.execute('''
            INSERT INTO ai_chat_messages (session_id, message_type, content, suggestions, timestamp)
            VALUES (%s, 'bot', %s, %s, NOW())
        ''', (session_id, bot_response, json.dumps(suggestions)))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error saving chat: {e}")
    finally:
        cursor.close()
        conn.close()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai_bot(request: ChatRequest, current_user = Depends(get_current_user)):
    """Chat with personalized AI bot"""
    
    if request.bot_type not in ["coach", "drop", "search", "rep", "cast"]:
        raise HTTPException(status_code=400, detail="Invalid bot type")
    
    user_id = current_user["id"]
    
    # Get user profile and content context
    user_profile = get_user_ai_profile(user_id)
    user_context = get_user_content_context(user_id)
    
    # Create personalized prompt
    personalized_prompt = create_personalized_prompt(request.bot_type, user_profile, user_context)
    
    # Get AI response
    ai_response, success = await get_ai_response(personalized_prompt, request.message)
    
    if not success:
        raise HTTPException(status_code=500, detail="AI service temporarily unavailable")
    
    # Generate suggestions
    suggestions = generate_personalized_suggestions(request.bot_type, user_profile)
    
    # Create or use existing session ID
    session_id = request.session_id or f"{user_id}_{request.bot_type}_{int(datetime.now().timestamp())}"
    
    # Save to database
    save_chat_message(session_id, user_id, request.bot_type, request.message, ai_response, suggestions)
    
    bot_names = {
        "coach": "RealmGPT Coach™",
        "drop": "RealmDrop™", 
        "search": "RealmSearch™",
        "rep": "RealmRep™",
        "cast": "RealmCast™"
    }
    
    return ChatResponse(
        response=ai_response,
        bot_name=bot_names[request.bot_type],
        suggestions=suggestions,
        session_id=session_id,
        user_context_used=bool(user_context and "No previous content" not in user_context)
    )

@router.post("/setup-profile")
async def setup_ai_profile(profile: UserAIProfile, current_user = Depends(get_current_user)):
    """Setup user's AI profile for personalized responses"""
    user_id = current_user["id"]
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_ai_profiles (user_id, industry, specialization, experience_level, goals, content_types, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                industry = EXCLUDED.industry,
                specialization = EXCLUDED.specialization,
                experience_level = EXCLUDED.experience_level,
                goals = EXCLUDED.goals,
                content_types = EXCLUDED.content_types,
                updated_at = NOW()
        ''', (user_id, profile.industry, profile.specialization, profile.experience_level, 
              json.dumps(profile.goals), json.dumps(profile.content_types)))
        conn.commit()
        
        return {"status": "success", "message": "AI profile updated successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@router.get("/available-bots")
async def get_available_bots():
    """Get list of all available AI bots"""
    return {
        "bots": [
            {
                "id": "coach",
                "name": "RealmGPT Coach™",
                "description": "AI coach to help speakers practice and improve",
                "icon": "🎯",
                "gradient": "from-blue-500 to-blue-700"
            },
            {
                "id": "drop", 
                "name": "RealmDrop™",
                "description": "Automatically repurpose long talks into short clips",
                "icon": "📱",
                "gradient": "from-purple-500 to-purple-700"
            },
            {
                "id": "search",
                "name": "RealmSearch™", 
                "description": "Help speakers find new speaking opportunities",
                "icon": "🔍",
                "gradient": "from-green-500 to-green-700"
            },
            {
                "id": "rep",
                "name": "RealmRep™",
                "description": "Manage reviews, bookings, and professional reputation",
                "icon": "⭐",
                "gradient": "from-yellow-500 to-yellow-700"
            },
            {
                "id": "cast",
                "name": "RealmCast™",
                "description": "Streaming + recording platform assistant",
                "icon": "📺",
                "gradient": "from-red-500 to-red-700"
            }
        ]
    }

@router.get("/chat-history")
async def get_chat_history(current_user = Depends(get_current_user)):
    """Get user's chat history"""
    user_id = current_user["id"]
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT s.session_id, s.bot_type, s.created_at,
                   m.message_type, m.content, m.suggestions, m.timestamp
            FROM ai_chat_sessions s
            JOIN ai_chat_messages m ON s.session_id = m.session_id
            WHERE s.user_id = %s
            ORDER BY s.created_at DESC, m.timestamp ASC
        ''', (user_id,))
        rows = cursor.fetchall()
        
        sessions = {}
        for row in rows:
            session_id, bot_type, session_created, msg_type, content, suggestions, msg_time = row
            
            if session_id not in sessions:
                sessions[session_id] = {
                    "session_id": session_id,
                    "bot_type": bot_type,
                    "created_at": session_created,
                    "messages": []
                }
            
            sessions[session_id]["messages"].append({
                "type": msg_type,
                "content": content,
                "suggestions": suggestions,
                "timestamp": msg_time
            })
        
        return {"status": "success", "sessions": list(sessions.values())}
    finally:
        cursor.close()
        conn.close()

