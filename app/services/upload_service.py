import os, tempfile
import whisper
import yt_dlp
import shutil
from app.utils.extractors import extract_text_from_pdf, extract_text_from_docx

# os.environ["PATH"] += os.pathsep + r"C:\Users\Yashal Rafique\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".webm"]
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov"]

async def handle_file_upload(file, fileType, youtubeUrl):
    print("in file controller")

    # ✅ Handle YouTube video transcription
    if fileType == "youtube" and youtubeUrl:
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
                'ffmpeg_location': shutil.which('ffmpeg'),  # ✅ Ye line add karo
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtubeUrl, download=True)
                    audio_path = os.path.join(tmpdir, f"downloaded_audio.mp3")

                # 🧠 Transcribe using Whisper
                model = whisper.load_model("base")
                print(f"Transcribing YouTube audio: {audio_path}")
                result = model.transcribe(audio_path)
                extracted_text = result["text"]

                return {
                    "status": "success",
                    "source": "youtube",
                    "url": youtubeUrl,
                    "text": extracted_text
                }

            except Exception as e:
                return {
                    "status": "error",
                    "source": "youtube",
                    "message": f"❌ Error downloading or processing YouTube audio: {str(e)}"
                }

    # ✅ Handle uploaded files
    if file:
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
                print(f"Transcribing file: {tmp_path}")
                result = model.transcribe(tmp_path)
                extracted_text = result["text"]
            else:
                extracted_text = "❌ Unsupported file format."
        except Exception as e:
            extracted_text = f"❌ Error while processing file: {str(e)}"
        finally:
            os.remove(tmp_path)

        return {
            "status": "success",
            "source": "file",
            "filename": file.filename,
            "fileType": fileType,
            "text": extracted_text
        }

    return {"status": "error", "message": "No file or URL provided"}
