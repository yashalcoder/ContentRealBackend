import os, tempfile
import whisper
import yt_dlp
import shutil
from app.utils.extractors import extract_text_from_pdf, extract_text_from_docx
from app.models.schemas import ContentCreate
import openai
from app.database import get_db_connection
from app.utils.JWT import get_current_user
from fastapi import  Depends
from dotenv import load_dotenv
load_dotenv()
print("openai key", os.getenv("OPENAI_API_KEY") )
os.environ["PATH"] += os.pathsep + os.getcwd()
openai.api_key = os.getenv("OPENAI_API_KEY")
# os.environ["PATH"] += os.pathsep + r"C:\Users\Yashal Rafique\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".webm"]
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov"]

def generate_post_with_ai(extracted_text: str) -> str:
    prompt = f"Write a short, engaging social media post based on this content:\n\n{extracted_text}"
    response =openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a creative content writer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

import openai
from typing import List
import json

def extract_content_topics_and_summary(extracted_text: str) -> tuple:
    """Extract topics and create summary using AI"""
    try:
        # Extract topics
        topics_prompt = f"""
        Analyze this content and extract 3-5 main topics/themes. Return only a JSON array of topics.
        Content: {extracted_text[:1000]}...
        
        Example format: ["leadership", "motivation", "business strategy"]
        """
        
        topics_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a content analyst. Return only valid JSON."},
                {"role": "user", "content": topics_prompt}
            ],
            max_tokens=100
        )
        
        topics = json.loads(topics_response.choices[0].message.content.strip())
        
        # Create summary
        summary_prompt = f"""
        Create a brief 2-3 sentence summary of this content highlighting the key insights:
        {extracted_text[:1500]}...
        """
        
        summary_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a content summarizer."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=150
        )
        
        summary = summary_response.choices[0].message.content.strip()
        
        return topics, summary
        
    except Exception as e:
        print(f"AI analysis error: {e}")
        return [], "Content analysis pending..."

def save_content_analytics(content_id: int, extracted_text: str):
    """Save AI analysis of content"""
    try:
        topics, summary = extract_content_topics_and_summary(extracted_text)
        
        # Simple sentiment analysis (you can use a proper sentiment library)
        sentiment_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible']
        
        text_lower = extracted_text.lower()
        positive_count = sum(1 for word in sentiment_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        sentiment_score = (positive_count - negative_count) / max(len(extracted_text.split()), 1)
        
        # Engagement prediction (simple heuristic)
        engagement_factors = {
            'questions': text_lower.count('?') * 0.1,
            'exclamations': text_lower.count('!') * 0.05,
            'call_to_action': len([word for word in ['subscribe', 'share', 'comment', 'like'] if word in text_lower]) * 0.15,
            'length': min(len(extracted_text) / 1000, 1) * 0.3
        }
        engagement_prediction = min(sum(engagement_factors.values()) + 0.5, 1.0)
        
        # Save to database
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO content_analytics (content_id, topics, sentiment_score, engagement_prediction, ai_summary)
                VALUES (%s, %s, %s, %s, %s)
            ''', (content_id, json.dumps(topics), sentiment_score, engagement_prediction, summary))
            conn.commit()
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        print(f"Error saving content analytics: {e}")

async def handle_file_upload(file, fileType, youtubeUrl,current_user):
    user_id = current_user["id"]  # JWT payload se ID
    print("in file controller")

    extracted_text = None
    source_url = None

    # ✅ Handle YouTube video transcription
    if fileType == "youtube" and youtubeUrl:
        source_url = youtubeUrl
        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(tmpdir, 'downloaded_audio.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'ffmpeg_location': shutil.which('ffmpeg'),
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.extract_info(youtubeUrl, download=True)
                    audio_path = os.path.join(tmpdir, f"downloaded_audio.mp3")
                model = whisper.load_model("base")
                result = model.transcribe(audio_path)
                extracted_text = result["text"]
            except Exception as e:
                return {"status": "error", "message": str(e)}

    # ✅ Handle uploaded files
    elif file:
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            if suffix == ".pdf":
                extracted_text = extract_text_from_pdf(tmp_path)
            elif suffix == ".docx":
                extracted_text = extract_text_from_docx(tmp_path)
            elif suffix == ".txt":
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
            elif suffix in AUDIO_EXTENSIONS + VIDEO_EXTENSIONS:
                model = whisper.load_model("base")
                result = model.transcribe(tmp_path)
                extracted_text = result["text"]
            else:
                extracted_text = "❌ Unsupported file format."
        finally:
            os.remove(tmp_path)

    if not extracted_text:
        return {"status": "error", "message": "No content extracted"}

    # 🧠 AI se post generate karo
    post_content = generate_post_with_ai(extracted_text)
    title = post_content[:50]  # title = first 50 chars

    # ✅ Database me insert karo
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO "Content"("FileType", "url", "PostContent","UserId","title")
            VALUES (%s, %s, %s,%s, %s)
            RETURNING id;
            ''',
            (fileType, source_url, post_content,user_id,title)
        )
        post_id = cursor.fetchone()[0]
        if post_id:
            try:
                save_content_analytics(post_id, extracted_text)
            except Exception as e:
                print(f"Background analytics failed: {e}")
        conn.commit()
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        conn.close()

    return {
        "status": "success",
        "extracted_text": extracted_text,
        "post_content": post_content,
        "post_id": post_id,
        "text": extracted_text 
    }


