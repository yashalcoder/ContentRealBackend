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


