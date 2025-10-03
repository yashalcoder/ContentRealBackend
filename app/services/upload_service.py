# app/services/upload_handler.py - Fixed version for localStorage approach

import os, tempfile
import whisper
import yt_dlp
import shutil
from app.utils.extractors import extract_text_from_pdf, extract_text_from_docx
from app.models.schemas import ContentCreate
import openai
import time
from app.database import get_db_connection
from app.utils.JWT import get_current_user
from fastapi import Depends
from dotenv import load_dotenv
load_dotenv()
import threading
import subprocess
import uuid
import json
from PIL import Image
import asyncio
from PIL import Image, ImageDraw, ImageFont

# ADD these imports at the top of your file
from PIL import Image, ImageDraw, ImageFont
print("openai key", os.getenv("OPENAI_API_KEY"))
os.environ["PATH"] += os.pathsep + os.getcwd()
openai.api_key = os.getenv("OPENAI_API_KEY")

WHISPER_MODEL = None
MODEL_LOCK = threading.Lock()

def get_whisper_model():
    global WHISPER_MODEL
    
    if WHISPER_MODEL is not None:
        return WHISPER_MODEL
    
    with MODEL_LOCK:
        if WHISPER_MODEL is not None:
            return WHISPER_MODEL
        
        try:
            print("Loading Whisper model (one-time setup)...")
            WHISPER_MODEL = whisper.load_model("base")
            
            # CRITICAL: Check if model actually loaded
            if WHISPER_MODEL is None:
                raise Exception("Whisper model loaded as None")
                
            print("✅ Whisper model loaded successfully")
            return WHISPER_MODEL
            
        except Exception as e:
            print(f"❌ Whisper model loading failed: {e}")
            WHISPER_MODEL = None  # Reset to None so it can retry
            raise e

AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".webm"]
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov"]
# Add this at the top of the file
# SOCIAL_PLATFORMS = [
#     {"platform": "instagram", "aspect_ratio": "9:16", "resolution": "1080x1920", "max_file_size_mb": 100, "caption_style": "bold_center"},
#     {"platform": "tiktok", "aspect_ratio": "9:16", "resolution": "1080x1920", "max_file_size_mb": 250, "caption_style": "colorful"},
#     {"platform": "linkedin", "aspect_ratio": "1:1", "resolution": "1080x1080", "max_file_size_mb": 200, "caption_style": "subtle_bottom"},
#     {"platform": "x", "aspect_ratio": "16:9", "resolution": "1920x1080", "max_file_size_mb": 250, "caption_style": "minimal"},
# ]


# # Use absolute paths for Replit
# CLIP_DIR = os.path.abspath("clips")
# THUMBNAILS_DIR = os.path.join(CLIP_DIR, "thumbnails")
# CAPTIONS_DIR = os.path.join(CLIP_DIR, "captions")
# Use /tmp for temporary storage (auto-cleanup on restart)
CLIP_DIR = os.getenv("CLIPS_DIR", "/tmp/clips")
THUMBNAILS_DIR = os.path.join(CLIP_DIR, "thumbnails")
CAPTIONS_DIR = os.path.join(CLIP_DIR, "captions")
# Create directories
os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(THUMBNAILS_DIR, exist_ok=True)
os.makedirs(CAPTIONS_DIR, exist_ok=True)

# In-memory storage for clips metadata (since you don't want database)
CLIPS_STORAGE = {}
async def generate_templates_for_clip(clip_path: str, clip_id: str, post_id: int, title: str) -> list:
    """Generate platform-specific templates in parallel"""
    
    async def process_platform(platform):
        platform_name = platform["platform"]
        aspect_ratio = platform["aspect_ratio"]
        resolution = platform.get("resolution")  # e.g. "1080x1920"
        
        output_name = f"{post_id}_{clip_id}_{platform_name}.mp4"
        output_path = os.path.join(CLIP_DIR, output_name)
        
        # Optimized video filters
        if aspect_ratio == "9:16":
            vf_filter = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"
        elif aspect_ratio == "1:1":
            vf_filter = "scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"
        elif aspect_ratio == "16:9":
            vf_filter = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"
        else:
            vf_filter = "scale=-2:720,format=yuv420p"

        cmd = [
        "ffmpeg", "-i", clip_path,
        "-vf", vf_filter,
        "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p",           # ADD THIS LINE
        "-profile:v", "high",            # ADD THIS LINE  
        "-preset", "fast", "-crf", "20", # Changed from 23 to 20
        "-movflags", "+faststart",
        "-y", output_path
        ]

        try:
            # Run in background thread
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            )

            if result.returncode == 0 and os.path.exists(output_path):
                # 🔹 File size check
                max_size = platform.get("max_file_size_mb")
                if max_size:
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    if file_size_mb > max_size:
                        print(f"⚠️ WARNING: {platform_name} video is {file_size_mb:.2f}MB (exceeds {max_size}MB limit)")
                    else:
                        print(f"✅ {platform_name} video size OK ({file_size_mb:.2f}MB <= {max_size}MB)")

                # 🔹 Resolution check
                if resolution:
                    # ffprobe se actual resolution read karein
                    probe_cmd = [
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0", output_path
                    ]
                    probe_result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if probe_result.returncode == 0:
                        width, height = probe_result.stdout.strip().split(",")
                        actual_res = f"{width}x{height}"
                        if actual_res != resolution:
                            print(f"⚠️ WARNING: {platform_name} resolution is {actual_res}, expected {resolution}")
                        else:
                            print(f"✅ {platform_name} resolution OK ({resolution})")

                return {
                    "platform": platform_name,
                    "aspect_ratio": aspect_ratio,
                    "file_path": output_path,
                    "caption_style": platform["caption_style"],
                    "filename": output_name
                }
            else:
                print(f"[{platform_name}] ffmpeg failed:\n{result.stderr}")
                return None
        except Exception as e:
            print(f"Error generating template for {platform_name}: {e}")
            return None
    
    # Process all platforms concurrently
    tasks = [process_platform(platform) for platform in SOCIAL_PLATFORMS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [r for r in results if r and not isinstance(r, Exception)]

async def detect_highlight_timestamps(transcript: str) -> list:
    """Ask OpenAI to pick highlight timestamps from transcript"""
    prompt = f"""
    From this transcript, extract 3-5 engaging highlight moments with precise timestamps.
    Focus on:
    - Key insights or revelations
    - Emotional peaks or exciting moments  
    - Quotable moments
    - Action sequences or important events
    
    Return as JSON in format:
    [{{"start": "00:01:10", "end": "00:01:35", "reason": "Key insight about success", "title": "Success Mindset"}}]

    Transcript:
    {transcript[:2500]}...
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a video editor AI. Extract the most engaging moments. Return JSON only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )
        highlights = json.loads(response.choices[0].message.content.strip())
        return highlights
    except Exception as e:
        print(f"Error detecting highlights: {e}")
        # Return fallback highlights
        return [
            {"start": "00:00:10", "end": "00:00:25", "reason": "Opening segment", "title": "Intro Clip"},
            {"start": "00:01:00", "end": "00:01:20", "reason": "Key moment", "title": "Highlight 1"},
            {"start": "00:02:00", "end": "00:02:30", "reason": "Important insight", "title": "Highlight 2"}
        ]

def generate_thumbnail(video_path: str, timestamp: str, output_path: str, title: str = "") -> bool:
    """Generate enhanced thumbnail with text overlay"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate base thumbnail
        temp_thumb = output_path.replace('.jpg', '_temp.jpg')
        cmd = [
            "ffmpeg", "-i", video_path,
            "-ss", timestamp,
            "-vframes", "1",
            "-q:v", "2",
            "-s", "1280x720",
            "-y", temp_thumb
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        
        if result.returncode != 0 or not os.path.exists(temp_thumb):
            return False
        
        # Enhance with text overlay using PIL
        with Image.open(temp_thumb) as img:
            img = img.resize((1280, 720), Image.Resampling.LANCZOS)
            enhanced = img.copy()
            draw = ImageDraw.Draw(enhanced)
            
            if title:
                # Add semi-transparent overlay for text readability
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 120))
                enhanced = Image.alpha_composite(img.convert('RGBA'), overlay)
                draw = ImageDraw.Draw(enhanced)
                
                # Try to load font, fallback to default
                try:
                    font_size = min(60, max(30, 800 // len(title)))
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                # Wrap text for long titles
                words = title.split()
                lines = []
                current_line = []
                
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    if bbox[2] - bbox[0] > 1000:  # Max width
                        if current_line:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                        else:
                            lines.append(word)
                    else:
                        current_line.append(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                # Draw text with shadow
                y_offset = (720 - len(lines) * font_size) // 2
                for i, line in enumerate(lines):
                    bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = bbox[2] - bbox[0]
                    x = (1280 - text_width) // 2
                    y = y_offset + i * font_size
                    
                    # Shadow
                    draw.text((x+3, y+3), line, font=font, fill=(0, 0, 0))
                    # Main text
                    draw.text((x, y), line, font=font, fill=(255, 255, 255))
            
            # Save final image
            enhanced.convert('RGB').save(output_path, 'JPEG', quality=90)
        
        # Clean up temp file
        os.remove(temp_thumb)
        return True
        
    except Exception as e:
        print(f"Error generating thumbnail: {e}")
        return False
def generate_captions(transcript: str, start_time: str, end_time: str, output_path: str) -> bool:
    """Generate properly formatted VTT captions for the clip segment"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert timestamps to seconds
        def time_to_seconds(time_str):
            parts = time_str.split(':')
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        
        start_seconds = time_to_seconds(start_time)
        end_seconds = time_to_seconds(end_time)
        duration = end_seconds - start_seconds
        
        # Format time for VTT (uses . instead of , for milliseconds)
        def seconds_to_vtt_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millisecs = int(round((seconds - int(seconds)) * 1000))
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

        # PROBLEM IDENTIFIED: You're using the ENTIRE transcript for ALL clips
        # SOLUTION: Extract only the relevant portion of the transcript
        
        # Estimate words per second (typical speech is 2-3 words per second)
        words_per_second = 2.5
        transcript_words = transcript.split()
        total_transcript_duration = len(transcript_words) / words_per_second
        
        # Calculate which portion of the transcript corresponds to this clip
        start_word_index = int((start_seconds / total_transcript_duration) * len(transcript_words))
        end_word_index = int((end_seconds / total_transcript_duration) * len(transcript_words))
        
        # Extract only the relevant portion of the transcript for this clip
        clip_transcript_words = transcript_words[start_word_index:end_word_index]
        
        # If we don't have enough words, use a broader range or fallback
        if len(clip_transcript_words) < 5:
            # Fallback: use a portion based on clip position in the full video
            fallback_start = max(0, start_word_index - 10)
            fallback_end = min(len(transcript_words), end_word_index + 10)
            clip_transcript_words = transcript_words[fallback_start:fallback_end]
        
        # Generate VTT content with clip-specific transcript
        vtt_content = ["WEBVTT", ""]  # REQUIRED VTT header
        current_time = 0
        segment_duration = 3  # 3 second segments
        
        # Estimate words per segment based on clip-specific transcript
        words_per_segment = max(3, len(clip_transcript_words) // max(1, int(duration / segment_duration)))
        
        segment_index = 1
        for i in range(0, len(clip_transcript_words), words_per_segment):
            if current_time >= duration:
                break
                
            segment_start = current_time
            segment_end = min(current_time + segment_duration, duration)
            
            start_vtt = seconds_to_vtt_time(segment_start)
            end_vtt = seconds_to_vtt_time(segment_end)
            
            # Get the specific text segment for this time range
            text_segment = " ".join(clip_transcript_words[i:i + words_per_segment])[:80]
            
            # Only add non-empty segments
            if text_segment.strip():
                vtt_content.append(f"{start_vtt} --> {end_vtt}")
                vtt_content.append(text_segment.strip())
                vtt_content.append("")
            
            segment_index += 1
            current_time = segment_end
        
        # If no segments were created, create a single segment with available text
        if len(vtt_content) <= 2:  # Only header exists
            text_for_clip = " ".join(clip_transcript_words)[:100] if clip_transcript_words else f"Clip segment {start_time} - {end_time}"
            vtt_content.append(f"00:00:00.000 --> {seconds_to_vtt_time(duration)}")
            vtt_content.append(text_for_clip)
            vtt_content.append("")
        
        # Write vtt file with proper encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vtt_content))
        
        print(f"✅ VTT captions created: {output_path}")
        print(f"📝 Used {len(clip_transcript_words)} words from transcript (indexes {start_word_index}-{end_word_index})")
        
        # Verify file was created and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"📝 VTT file size: {os.path.getsize(output_path)} bytes")
            return True
        else:
            print(f"❌ VTT file creation failed or empty")
            return False
        
    except Exception as e:
        print(f"❌ Error generating VTT captions: {e}")
        return False



def cut_clip_with_extras(input_video: str, start: str, end: str, transcript: str = "", title: str = "",post_id:int=None) -> dict:
    """Cut a clip from video and generate thumbnail + captions"""
    
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    
    clip_id = str(uuid.uuid4())
    safe_title = title.replace(" ", "_").replace("/", "_")[:30] or f"clip{clip_id}"
    clip_name = f"{post_id}_{clip_id}_{safe_title}.mp4"
    thumbnail_name = f"{post_id}_{clip_id}_{safe_title}.jpg"
    caption_name = f"{post_id}_{clip_id}_{safe_title}.vtt"

    clip_path = os.path.join(CLIP_DIR, clip_name)
    thumbnail_path = os.path.join(THUMBNAILS_DIR, thumbnail_name)
    caption_path = os.path.join(CAPTIONS_DIR, caption_name)
    
    try:
        print(f"🎬 Cutting clip: {start} to {end}")
        print(f"Input: {input_video}")
        print(f"Output: {clip_path}")
        
        # 1. Cut the video clip
        cmd = [
            "ffmpeg", "-i", input_video,
            "-ss", start, "-to", end,
            "-c:v", "libx264", "-c:a", "aac",
            "-preset", "fast", "-crf", "23",
            "-avoid_negative_ts", "make_zero",
            "-y", clip_path
        ]
        
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            timeout=180
        )
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
            raise RuntimeError(f"Generated clip is empty or missing: {clip_path}")
        
        # 2. Generate thumbnail
        def time_to_seconds(time_str):
            parts = time_str.split(':')
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        
        start_seconds = time_to_seconds(start)
        end_seconds = time_to_seconds(end)
        mid_seconds = (start_seconds + end_seconds) / 2
        
        def seconds_to_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h:02d}:{m:02d}:{s:06.3f}"
        
        mid_timestamp = seconds_to_time(mid_seconds)
        thumbnail_generated = generate_thumbnail(input_video, mid_timestamp, thumbnail_path)
        
        # 3. Generate captions
        captions_generated = False
        if transcript:
            

            captions_generated = generate_captions(transcript, start, end, caption_path)
        
        duration = end_seconds - start_seconds
        file_size = os.path.getsize(clip_path)
        
        print(f"✅ Clip created successfully: {clip_path}")
        print(f"📸 Thumbnail: {'✅' if thumbnail_generated else '❌'}")
        print(f"📝 Captions: {'✅' if captions_generated else '❌'}")
        print(f"📏 Size: {file_size / 1024 / 1024:.2f} MB")
        
        return {
            "id": clip_id,
            "clip": clip_path,
            "thumbnail": thumbnail_path if thumbnail_generated else None,
            "captions": caption_path if captions_generated else None,
            "start": start,
            "end": end,
            "duration": duration,
            "title": title,
            "reason": title,
            "file_size": file_size,
            "filename": clip_name,
            "exists": True
        }
        
    except Exception as e:
        # Clean up any partial files
        for path in [clip_path, thumbnail_path, caption_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        print(f"❌ Error in cut_clip_with_extras: {e}")
        raise

def generate_post_with_ai(extracted_text: str,no_of_posts:int) -> str:
    try:
        print("noi of psots",no_of_posts)
       
        prompt = f"""Write {no_of_posts} short, engaging social media posts based on this content.

        Format your response EXACTLY like this:
        Post 1:
        [first post content here]

        Post 2:
        [second post content here]

        Post 3:
        [third post content here]

        And so on for all {no_of_posts} posts.

        Content to base posts on:
        {extracted_text[:1000]}

        Remember: Each post must start with "Post X:" on its own line."""
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative content writer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        print(response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating post: {e}")
        return "AI-generated content based on uploaded material."
# Replace your caption generation function with this direct approach:

def generate_accurate_captions_from_clip(clip_video_path: str, output_path: str) -> bool:
    """Generate captions by loading fresh Whisper model each time"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Transcribing clip: {os.path.basename(clip_video_path)}")
        
        # Load fresh model each time to avoid corruption
        import whisper
        model = whisper.load_model("base")
        
        # Simple transcription without advanced parameters
        result = model.transcribe(clip_video_path)
        
        # Clean up model immediately
        del model
        
        # Generate VTT from segments
        def seconds_to_vtt_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millisecs = int(round((seconds - int(seconds)) * 1000))
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
        
        vtt_content = ["WEBVTT", ""]
        segments = result.get("segments", [])
        
        if not segments:
            # Fallback: use the full text
            text = result.get("text", "").strip()
            if text:
                vtt_content.extend(["00:00:00.000 --> 00:00:10.000", text, ""])
        else:
            for segment in segments:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", start_time + 3)
                text = segment.get("text", "").strip()
                
                if text:
                    start_vtt = seconds_to_vtt_time(start_time)
                    end_vtt = seconds_to_vtt_time(end_time)
                    vtt_content.extend([f"{start_vtt} --> {end_vtt}", text, ""])
        
        # Write VTT file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vtt_content))
        
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        
    except Exception as e:
        print(f"Caption generation failed: {e}")
        return False
async def cut_clip_with_perfect_sync(input_video: str, start: str, end: str, transcript: str = "", title: str = "", post_id: int = None) -> dict:
    """Optimized clip cutting with parallel processing"""
    
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    
    clip_id = str(uuid.uuid4())
    safe_title = title.replace(" ", "_").replace("/", "_")[:30] or f"clip{clip_id}"
    
    clip_name = f"{post_id}_{clip_id}_{safe_title}.mp4"
    thumbnail_name = f"{post_id}_{clip_id}_{safe_title}.jpg"
    caption_name = f"{post_id}_{clip_id}_{safe_title}.vtt"

    clip_path = os.path.join(CLIP_DIR, clip_name)
    thumbnail_path = os.path.join(THUMBNAILS_DIR, thumbnail_name)
    caption_path = os.path.join(CAPTIONS_DIR, caption_name)
    
    try:
        # Step 1: Cut video with optimized settings
        cmd = [
            "ffmpeg", "-i", input_video,
            "-ss", start, "-to", end,
            "-c:v", "libx264", "-c:a", "aac",
            "-pix_fmt", "yuv420p",           # ADD THIS LINE
            "-profile:v", "high",            # ADD THIS LINE
            "-preset", "ultrafast",
            "-crf", "23",                    # Changed from 25 to 23
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            "-threads", "0",
            "-y", clip_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        # Step 2: Generate assets in parallel
        async def generate_thumbnail_task():
            def time_to_seconds(time_str):
                parts = time_str.split(':')
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            
            start_seconds = time_to_seconds(start)
            end_seconds = time_to_seconds(end)
            mid_seconds = (start_seconds + end_seconds) / 2
            
            def seconds_to_time(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = seconds % 60
                return f"{h:02d}:{m:02d}:{s:06.3f}"
            
            mid_timestamp = seconds_to_time(mid_seconds)
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, generate_thumbnail, input_video, mid_timestamp, thumbnail_path, title
            )
        
        async def generate_captions_task():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, generate_accurate_captions_from_clip, clip_path, caption_path
            )
        
        # Run thumbnail and caption generation in parallel
        thumbnail_success, captions_success = await asyncio.gather(
            generate_thumbnail_task(),
            generate_captions_task(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(thumbnail_success, Exception):
            thumbnail_success = False
        if isinstance(captions_success, Exception):
            captions_success = False
        
        # Step 3: Generate templates
        templates = await generate_templates_for_clip(clip_path, clip_id, post_id, title)
        
        # Calculate metrics
        def time_to_seconds(time_str):
            parts = time_str.split(':')
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        
        duration = time_to_seconds(end) - time_to_seconds(start)
        file_size = os.path.getsize(clip_path)
        
        return {
            "id": clip_id,
            "clip": clip_path,
            "thumbnail": thumbnail_path if thumbnail_success else None,
            "captions": caption_path if captions_success else None,
            "start": start,
            "end": end,
            "duration": duration,
            "title": title,
            "reason": title,
            "file_size": file_size,
            "filename": clip_name,
            "exists": True,
            "clip_filename": clip_name,
            "thumbnail_filename": thumbnail_name if thumbnail_success else None,
            "captions_filename": caption_name if captions_success else None,
            "templates": templates
        }
        
    except Exception as e:
        # Clean up partial files
        for path in [clip_path, thumbnail_path, caption_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        raise
async def process_clips_parallel(video_file_path: str, highlights: list, post_id: int, extracted_text: str) -> list:
    """Process all clips in parallel with limited concurrency"""
    
    # Limit concurrent processing to avoid resource exhaustion
    semaphore = asyncio.Semaphore(3)  # Process max 3 clips at once
    
    async def process_single_clip(highlight):
        async with semaphore:
            try:
                return await cut_clip_with_perfect_sync(
                    video_file_path,
                    highlight["start"],
                    highlight["end"],
                    extracted_text,
                    highlight.get("reason", "Highlight"),
                    post_id
                )
            except Exception as e:
                print(f"Failed to process clip {highlight['start']}-{highlight['end']}: {e}")
                return None
    
    # Create tasks for all clips
    tasks = [process_single_clip(h) for h in highlights]
    
    # Process all clips concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results
    successful_clips = [r for r in results if r and not isinstance(r, Exception)]
    
    print(f"Processed {len(successful_clips)} clips successfully out of {len(highlights)}")
    return successful_clips
def store_clips_in_memory(content_id: int, clips_data: list):
    """Store clips metadata in memory instead of database"""
    CLIPS_STORAGE[content_id] = {
        "clips": clips_data,
        "created_at": time.time()
    }
    print(f"📦 Stored {len(clips_data)} clips for content {content_id} in memory")

def get_clips_from_memory(content_id: int) -> dict:
    """Retrieve clips from memory storage"""
    return CLIPS_STORAGE.get(content_id, {"clips": []})

# async def handle_file_upload(file, fileType, youtubeUrl, current_user):
#     user_id = current_user["id"]
#     print("🚀 Starting file upload processing")

#     extracted_text = None
#     source_url = None
#     video_file_path = None
#     is_video_content = False

#     # Handle YouTube video transcription
    
#     # Handle YouTube video transcription
#     if fileType == "youtube" and youtubeUrl:
#         source_url = youtubeUrl
#         temp_dir = tempfile.mkdtemp(prefix="youtube_download_")
        
#         try:
#             print(f"🎬 Processing YouTube URL: {youtubeUrl}")
            
#             # Enhanced format options with better fallbacks
#             format_strategies = [
#                 # Strategy 1: Try best available with height limit
#                 {
#                     'format': 'best[height<=720][ext=mp4]/best[height<=480][ext=mp4]/best[ext=mp4]/best',
#                     'description': 'Best quality MP4 (720p or lower)'
#                 },
#                 # Strategy 2: Audio + Video merge
#                 {
#                     'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[ext=mp4]/best',
#                     'description': 'Merge best video + audio'
#                 },
#                 # Strategy 3: Any available format
#                 {
#                     'format': 'worst[height>=360]/worst',
#                     'description': 'Lowest quality available'
#                 },
#                 # Strategy 4: Audio only fallback
#                 {
#                     'format': 'bestaudio[ext=m4a]/bestaudio/best[acodec^=mp4a]',
#                     'description': 'Audio only'
#                 },
#                 # Strategy 5: Generic best
#                 {
#                     'format': 'best',
#                     'description': 'Best available format'
#                 }
#             ]
            
#             download_success = False
            
#             for i, strategy in enumerate(format_strategies):
#                 try:
#                     print(f"📥 Attempting strategy {i+1}: {strategy['description']}")
                    
#                     # Enhanced yt-dlp options
#                     ydl_opts = {
#                         'format': strategy['format'],
#                         'outtmpl': os.path.join(temp_dir, f'video_attempt_{i}.%(ext)s'),
#                         'quiet': False,
#                         'no_warnings': False,
#                         'ignoreerrors': False,
#                         'ffmpeg_location': shutil.which('ffmpeg'),
                        
#                         # Network and retry options
#                         'retries': 5,
#                         'fragment_retries': 5,
#                         'socket_timeout': 60,
#                         'http_chunk_size': 10485760,  # 10MB chunks
                        
#                         # User agent rotation
#                         'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        
#                         # Additional options to bypass restrictions
#                         'extractor_args': {
#                             'youtube': {
#                                 'player_client': ['android', 'web'],
#                                 'player_skip': ['webpage', 'configs'],
#                             }
#                         },
                        
#                         # Headers to appear more like a browser
#                         'http_headers': {
#                             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
#                             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#                             'Accept-Language': 'en-us,en;q=0.5',
#                             'Accept-Encoding': 'gzip,deflate',
#                             'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
#                             'Connection': 'keep-alive',
#                         },
                        
#                         # Disable metadata writing to speed up
#                         'writeinfojson': False,
#                         'writesubtitles': False,
#                         'writeautomaticsub': False,
#                         'writethumbnail': False,
#                         'writedescription': False,
#                     }
                    
#                     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                         # First, try to extract info to check if video is accessible
#                         try:
#                             info = ydl.extract_info(youtubeUrl, download=False)
#                             print(f"✅ Video info extracted: {info.get('title', 'Unknown')} - Duration: {info.get('duration', 'Unknown')}s")
#                         except Exception as info_error:
#                             print(f"⚠️ Could not extract video info: {info_error}")
#                             # Continue anyway, sometimes download works even if info extraction fails
                        
#                         # Now attempt download
#                         ydl.download([youtubeUrl])
                    
#                     # Check if download was successful
#                     video_files = [f for f in os.listdir(temp_dir) if f.startswith(f'video_attempt_{i}.')]
#                     if video_files:
#                         video_file_path = os.path.join(temp_dir, video_files[0])
#                         if os.path.exists(video_file_path) and os.path.getsize(video_file_path) > 1024:  # At least 1KB
#                             # Check if it's video or audio
#                             file_ext = os.path.splitext(video_files[0])[1].lower()
#                             is_video_content = file_ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm']
                            
#                             download_success = True
#                             print(f"✅ {'Video' if is_video_content else 'Audio'} downloaded successfully: {video_files[0]}")
#                             print(f"📏 File size: {os.path.getsize(video_file_path) / 1024 / 1024:.2f} MB")
#                             break
#                         else:
#                             print(f"❌ Downloaded file is empty or too small")
                    
#                 except yt_dlp.utils.DownloadError as dl_error:
#                     print(f"❌ Download error for strategy {i+1}: {dl_error}")
#                     continue
#                 except Exception as format_error:
#                     print(f"❌ Strategy {i+1} failed: {format_error}")
#                     continue
            
#             if not download_success:
#                 # Final fallback: Try with different extractor args
#                 try:
#                     print("🔄 Final attempt with alternative extractor settings...")
                    
#                     final_opts = {
#                         'format': 'worst',  # Just get anything that works
#                         'outtmpl': os.path.join(temp_dir, 'final_attempt.%(ext)s'),
#                         'quiet': True,
#                         'no_warnings': True,
#                         'ignoreerrors': True,
#                         'ffmpeg_location': shutil.which('ffmpeg'),
#                         'retries': 3,
#                         'socket_timeout': 30,
                        
#                         # Try different player client
#                         'extractor_args': {
#                             'youtube': {
#                                 'player_client': ['android_creator', 'android_music', 'android_embedded'],
#                                 'player_skip': ['configs'],
#                             }
#                         }
#                     }
                    
#                     with yt_dlp.YoutubeDL(final_opts) as ydl:
#                         ydl.download([youtubeUrl])
                    
#                     final_files = [f for f in os.listdir(temp_dir) if f.startswith('final_attempt.')]
#                     if final_files:
#                         video_file_path = os.path.join(temp_dir, final_files[0])
#                         if os.path.exists(video_file_path) and os.path.getsize(video_file_path) > 1024:
#                             download_success = True
#                             is_video_content = os.path.splitext(final_files[0])[1].lower() in ['.mp4', '.mkv', '.avi', '.mov', '.webm']
#                             print(f"✅ Final attempt successful: {final_files[0]}")
                        
#                 except Exception as final_error:
#                     print(f"❌ Final attempt also failed: {final_error}")
            
#             if not download_success:
#                 error_msg = """
#                 All download methods failed. This could be due to:
#                 1. Video is age-restricted or requires sign-in
#                 2. Video is private or region-blocked
#                 3. YouTube has updated their protection measures
#                 4. Network connectivity issues
#                 5. The video URL is invalid or video was deleted
                
#                 Suggestions:
#                 - Try a different YouTube video
#                 - Check if the video is publicly accessible
#                 - Ensure the URL is correct and complete
#                 - Try again in a few minutes (temporary restrictions)
#                 """
#                 raise Exception(error_msg)
#             # In your YouTube processing section, replace the transcription part:

#             print("Starting transcription...")
#             try:
#                 import whisper
#                 model = whisper.load_model("base")
#                 result = model.transcribe(video_file_path)
#                 extracted_text = result["text"].strip()
                
#                 # Clean up model immediately
#                 del model
                
#                 if extracted_text and len(extracted_text) >= 10:
#                     print(f"Transcription completed. Length: {len(extracted_text)} chars")
#                 else:
#                     raise Exception("Transcription produced no meaningful text")
                    
#             except Exception as transcribe_error:
#                 print(f"Transcription failed: {transcribe_error}")
#                 extracted_text = f"Audio transcription failed: {str(transcribe_error)}"
#         except Exception as e:
#             print(f"❌ YouTube processing failed: {e}")
#             # Clean up
#             if 'temp_dir' in locals():
#                 shutil.rmtree(temp_dir, ignore_errors=True)
#             return {"status": "error", "message": f"YouTube processing failed: {str(e)}"}

#     # ... rest of your existing code for other file types ...
#     # Handle uploaded files
#     elif file:
#         suffix = os.path.splitext(file.filename)[1].lower()
#         temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
        
#         try:
#             with os.fdopen(temp_fd, 'wb') as tmp_file:
#                 tmp_file.write(await file.read())
            
#             video_file_path = temp_path
            
#             if suffix in VIDEO_EXTENSIONS:
#                 is_video_content = True
            
#             # Extract text based on file type
#             if suffix == ".pdf":
#                 extracted_text = extract_text_from_pdf(temp_path)
#             elif suffix == ".docx":
#                 extracted_text = extract_text_from_docx(temp_path)
#             elif suffix == ".txt":
#                 with open(temp_path, 'r', encoding='utf-8') as f:
#                     extracted_text = f.read()
#             elif suffix in AUDIO_EXTENSIONS + VIDEO_EXTENSIONS:
#                 model = get_whisper_model()
#                 result = model.transcribe(temp_path)
#                 extracted_text = result["text"]
#             else:
#                 extracted_text = "❌ Unsupported file format."
                
#         except Exception as e:
#             try:
#                 os.unlink(temp_path)
#             except:
#                 pass
#             return {"status": "error", "message": str(e)}

#     if not extracted_text:
#         return {"status": "error", "message": "No content extracted"}

#     # Generate AI post
#     post_content = generate_post_with_ai(extracted_text)
#     title = post_content[:50]

#     # Save to database (minimal data)
#     conn = get_db_connection()
#     try:
#         cursor = conn.cursor()
#         cursor.execute(
#             '''
#             INSERT INTO "Content"("FileType", "url", "PostContent","UserId","title")
#             VALUES (%s, %s, %s,%s, %s)
#             RETURNING id;
#             ''',
#             (fileType, source_url, post_content, user_id, title)
#         )
#         post_id = cursor.fetchone()[0]
#         conn.commit()
        
#         clip_paths = []
#         highlights = []
        
#         if post_id and is_video_content and video_file_path:
#             try:
#                 print(f"🎬 Generating clips from: {video_file_path}")
                
#                 # Detect highlights
#                 highlights = await detect_highlight_timestamps(extracted_text)
#                 print(f"📝 Found {len(highlights)} highlights: {[h.get('reason', 'N/A') for h in highlights]}")
                
#                 # PARALLEL CLIP GENERATION - Much faster!
#                 start_time = time.time()
#                 clip_paths = await process_clips_parallel(
#                     video_file_path, 
#                     highlights, 
#                     post_id, 
#                     extracted_text
#                 )
                
#                 end_time = time.time()
#                 print(f"Generated {len(clip_paths)} clips in {end_time - start_time:.1f} seconds")

                
#                 # Store clips in memory instead of database
#                 if clip_paths:
#                     store_clips_in_memory(post_id, clip_paths)
#                     print(f"✅ Stored {len(clip_paths)} clips in memory for content {post_id}")

#             except Exception as e:
#                 print(f"Background clip generation failed: {e}")
        
#         # Clean up temporary files
#         try:
#             if 'temp_dir' in locals():
#                 shutil.rmtree(temp_dir, ignore_errors=True)
#             if 'temp_path' in locals() and os.path.exists(temp_path):
#                 os.unlink(temp_path)
#         except:
#             pass
                
#     except Exception as e:
#         conn.rollback()
#         return {"status": "error", "message": str(e)}
#     finally:
#         cursor.close()
#         conn.close()

#     return {
#         "status": "success",
#         "extracted_text": extracted_text,
#         "post_content": post_content,
#         "post_id": post_id,
#         "text": extracted_text,
#         "highlights": highlights,
#         "clips": clip_paths,
#         "video_file_path": video_file_path if is_video_content else None,
#         "clips_count": len(clip_paths)
#     }
# Updated SOCIAL_PLATFORMS - removed file size limits, keep only format specs
SOCIAL_PLATFORMS = [
    {"platform": "instagram", "aspect_ratio": "9:16", "resolution": "1080x1920", "caption_style": "bold_center"},
    {"platform": "tiktok", "aspect_ratio": "9:16", "resolution": "1080x1920", "caption_style": "colorful", "supported_formats": ["MP4", "MOV"]},
    {"platform": "linkedin", "aspect_ratio": "1:1", "resolution": "1080x1080", "caption_style": "subtle_bottom"},
    {"platform": "x", "aspect_ratio": "16:9", "resolution": "1920x1080", "caption_style": "minimal"},
]

# File size limits for different platforms (in MB)
PLATFORM_LIMITS = {
    "tiktok": {"android": 72, "ios": 287, "desktop": 1000},
    "instagram": {"general": 100},
    "linkedin": {"general": 200},
    "x": {"general": 250}
}

def validate_file_for_upload(file_path: str, file_format: str, target_platform: str = None) -> dict:
    """Validate file size and format before processing"""
    
    if not os.path.exists(file_path):
        return {"valid": False, "error": "File not found"}
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    file_ext = os.path.splitext(file_path)[1].upper().replace('.', '')
    
    validation_result = {
        "valid": True,
        "file_size_mb": file_size_mb,
        "format": file_ext,
        "warnings": [],
        "platform_compatibility": {}
    }
    
    # Check format compatibility
    if target_platform == "tiktok":
        if file_ext not in ["MP4", "MOV"]:
            validation_result["warnings"].append(f"TikTok prefers MP4/MOV format, got {file_ext}")
    
    # Check file size limits for all platforms
    for platform, limits in PLATFORM_LIMITS.items():
        if "general" in limits:
            max_size = limits["general"]
            validation_result["platform_compatibility"][platform] = {
                "compatible": file_size_mb <= max_size,
                "max_size": max_size,
                "current_size": file_size_mb
            }
        else:
            # TikTok has multiple limits
            compatibility = {}
            for device, max_size in limits.items():
                compatibility[device] = file_size_mb <= max_size
            
            validation_result["platform_compatibility"][platform] = {
                "android": compatibility.get("android", False),
                "ios": compatibility.get("ios", False), 
                "desktop": compatibility.get("desktop", False),
                "limits": limits,
                "current_size": file_size_mb
            }
    
    return validation_result

async def generate_templates_with_embedded_captions(clip_path: str, clip_id: str, post_id: int, title: str, captions_path: str = None) -> list:
    """Generate platform-specific templates with embedded captions"""
    
    async def process_platform(platform):
        platform_name = platform["platform"]
        aspect_ratio = platform["aspect_ratio"]
        resolution = platform.get("resolution")
        
        output_name = f"{post_id}_{clip_id}_{platform_name}.mp4"
        output_path = os.path.join(CLIP_DIR, output_name)
        
        # Build video filters
        if aspect_ratio == "9:16":
            vf_filter = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black"
        elif aspect_ratio == "1:1":
            vf_filter = "scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2:color=black"
        elif aspect_ratio == "16:9":
            vf_filter = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"
        else:
            vf_filter = "scale=-2:720"

        # Add subtitle filter if captions exist
        if captions_path and os.path.exists(captions_path):
            # Escape the path for FFmpeg
            escaped_path = captions_path.replace('\\', '\\\\').replace(':', '\\:')
            subtitle_filter = f"subtitles='{escaped_path}':force_style='FontSize=24,PrimaryColour=&H00ffffff,OutlineColour=&H00000000,Outline=2,Shadow=1'"
            vf_filter = f"{vf_filter},{subtitle_filter}"
            print(f"Adding captions to {platform_name}: {captions_path}")

        cmd = [
            "ffmpeg", "-i", clip_path,
            "-vf", vf_filter,
            "-c:v", "libx264", "-c:a", "aac",
            "-preset", "fast", "-crf", "23",
            "-movflags", "+faststart",
            "-y", output_path
        ]

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            )

            if result.returncode == 0 and os.path.exists(output_path):
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                
                # Validate the generated file
                validation = validate_file_for_upload(output_path, "mp4", platform_name)
                
                return {
                    "platform": platform_name,
                    "aspect_ratio": aspect_ratio,
                    "file_path": output_path,
                    "caption_style": platform["caption_style"],
                    "filename": output_name,
                    "file_size_mb": file_size_mb,
                    "validation": validation,
                    "has_embedded_captions": captions_path is not None
                }
            else:
                print(f"[{platform_name}] FFmpeg failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error generating template for {platform_name}: {e}")
            return None
    
    tasks = [process_platform(platform) for platform in SOCIAL_PLATFORMS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [r for r in results if r and not isinstance(r, Exception)]

async def cut_clip_with_embedded_captions(input_video: str, start: str, end: str, transcript: str = "", title: str = "", post_id: int = None) -> dict:
    """Cut clip and generate templates with embedded captions"""
    
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    
    clip_id = str(uuid.uuid4())
    safe_title = title.replace(" ", "_").replace("/", "_")[:30] or f"clip{clip_id}"
    
    clip_name = f"{post_id}_{clip_id}_{safe_title}.mp4"
    thumbnail_name = f"{post_id}_{clip_id}_{safe_title}.jpg"
    caption_name = f"{post_id}_{clip_id}_{safe_title}.vtt"

    clip_path = os.path.join(CLIP_DIR, clip_name)
    thumbnail_path = os.path.join(THUMBNAILS_DIR, thumbnail_name)
    caption_path = os.path.join(CAPTIONS_DIR, caption_name)
    
    try:
        # Step 1: Cut video clip
        cmd = [
                "ffmpeg", "-i", input_video,
                "-ss", start, "-to", end,
                "-c:v", "libx264", "-c:a", "aac",
                "-pix_fmt", "yuv420p",           # ADD THIS LINE
                "-profile:v", "high",            # ADD THIS LINE
                "-preset", "fast", "-crf", "20", # Changed from 23 to 20
                "-avoid_negative_ts", "make_zero",
                "-threads", "0",
                "-y", clip_path
            ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        # Validate the cut clip
        clip_validation = validate_file_for_upload(clip_path, "mp4")
        
        # Step 2: Generate assets in parallel
        async def generate_thumbnail_task():
            def time_to_seconds(time_str):
                parts = time_str.split(':')
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            
            start_seconds = time_to_seconds(start)
            end_seconds = time_to_seconds(end)
            mid_seconds = (start_seconds + end_seconds) / 2
            
            def seconds_to_time(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = seconds % 60
                return f"{h:02d}:{m:02d}:{s:06.3f}"
            
            mid_timestamp = seconds_to_time(mid_seconds)
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, generate_thumbnail, input_video, mid_timestamp, thumbnail_path, title
            )
        
        async def generate_captions_task():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, generate_accurate_captions_from_clip, clip_path, caption_path
            )
        
        # Generate thumbnail and captions in parallel
        thumbnail_success, captions_success = await asyncio.gather(
            generate_thumbnail_task(),
            generate_captions_task(),
            return_exceptions=True
        )
        
        if isinstance(thumbnail_success, Exception):
            thumbnail_success = False
        if isinstance(captions_success, Exception):
            captions_success = False
        
        # Step 3: Generate templates with embedded captions
        captions_file = caption_path if captions_success else None
        templates = await generate_templates_with_embedded_captions(
            clip_path, clip_id, post_id, title, captions_file
        )
        
        # Calculate metrics
        def time_to_seconds(time_str):
            parts = time_str.split(':')
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        
        duration = time_to_seconds(end) - time_to_seconds(start)
        file_size = os.path.getsize(clip_path)
        
        return {
            "id": clip_id,
            "clip": clip_path,
            "thumbnail": thumbnail_path if thumbnail_success else None,
            "captions": caption_path if captions_success else None,
            "start": start,
            "end": end,
            "duration": duration,
            "title": title,
            "reason": title,
            "file_size": file_size,
            "filename": clip_name,
            "exists": True,
            "clip_filename": clip_name,
            "thumbnail_filename": thumbnail_name if thumbnail_success else None,
            "captions_filename": caption_name if captions_success else None,
            "templates": templates,
            "validation": clip_validation,
            "has_embedded_captions": captions_success
        }
        
    except Exception as e:
        # Clean up partial files
        for path in [clip_path, thumbnail_path, caption_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        raise

async def process_clips_with_validation(video_file_path: str, highlights: list, post_id: int, extracted_text: str) -> list:
    """Process clips with file validation"""
    
    # First validate the source video
    source_validation = validate_file_for_upload(video_file_path, 
                                                os.path.splitext(video_file_path)[1].replace('.', ''))
    print(f"Source video validation: {source_validation}")
    
    if not source_validation["valid"]:
        print(f"Source video validation failed: {source_validation.get('error', 'Unknown error')}")
        return []
    
    semaphore = asyncio.Semaphore(3)
    
    async def process_single_clip(highlight):
        async with semaphore:
            try:
                return await cut_clip_with_embedded_captions(
                    video_file_path,
                    highlight["start"],
                    highlight["end"],
                    extracted_text,
                    highlight.get("reason", "Highlight"),
                    post_id
                )
            except Exception as e:
                print(f"Failed to process clip {highlight['start']}-{highlight['end']}: {e}")
                return None
    
    tasks = [process_single_clip(h) for h in highlights]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_clips = [r for r in results if r and not isinstance(r, Exception)]
    
    # Print validation summary
    print(f"\n=== FILE VALIDATION SUMMARY ===")
    for clip in successful_clips:
        validation = clip.get("validation", {})
        print(f"Clip: {clip['filename']}")
        print(f"Size: {validation.get('file_size_mb', 0):.2f}MB")
        
        compatibility = validation.get('platform_compatibility', {})
        for platform, compat in compatibility.items():
            if platform == "tiktok":
                android_ok = compat.get('android', False)
                ios_ok = compat.get('ios', False)
                desktop_ok = compat.get('desktop', False)
                print(f"  TikTok: Android({android_ok}) iOS({ios_ok}) Desktop({desktop_ok})")
            else:
                is_compat = compat.get('compatible', False)
                print(f"  {platform.title()}: {'✓' if is_compat else '✗'}")
        print()
    
    return successful_clips
async def handle_file_upload(file, fileType, youtubeUrl, current_user,no_of_posts):
    
    print("num of posts in handle (after conversion):", no_of_posts)
    print("type:", type(no_of_posts))
    user_id = current_user["id"]
    print("🚀 Starting file upload processing with validation")

    extracted_text = None
    source_url = None
    video_file_path = None
    is_video_content = False

    # Handle YouTube video transcription
    # Enhanced YouTube download section with bot protection bypass
# Replace your YouTube section in handle_file_upload() with this:

    if fileType == "youtube" and youtubeUrl:
        source_url = youtubeUrl
        temp_dir = tempfile.mkdtemp(prefix="youtube_download_")
        
        try:
            print(f"🎬 Processing YouTube URL: {youtubeUrl}")
            
            # Base options that work across all strategies
            base_opts = {
                'quiet': False,
                'no_warnings': False,
                'ignoreerrors': False,
                'ffmpeg_location': shutil.which('ffmpeg'),
                
                # Network and retry options
                'retries': 10,
                'fragment_retries': 10,
                'socket_timeout': 60,
                'http_chunk_size': 10485760,
                
                # CRITICAL: Use cookies file
                'cookiefile': '/root/cookies.txt',  # Update this path!
                
                # Enhanced user agent
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                
                # Enhanced extractor args with more client options
                'extractor_args': {
                    'youtube': {
                        'player_client': [
                            'android_creator',
                            'android_vr', 
                            'android_music',
                            'android_embedded',
                            'android',
                            'web',
                            'mweb',
                            'tv_embedded'
                        ],
                        'player_skip': ['webpage', 'configs', 'js'],
                        'skip': ['hls', 'dash'],
                    }
                },
                
                # Better headers
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip,deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Cache-Control': 'max-age=0',
                },
                
                # Disable unnecessary features
                'writeinfojson': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
                'writethumbnail': False,
                'writedescription': False,
                
                # Add random sleep to avoid rate limiting
                'sleep_interval': 2,
                'max_sleep_interval': 5,
            }
            
            # Format strategies with adjusted priorities
            format_strategies = [
                {
                    'format': 'best[height<=720][ext=mp4]/best[height<=480][ext=mp4]/best[ext=mp4]',
                    'description': 'Best quality MP4 (720p or lower)',
                    'merge_output_format': 'mp4'
                },
                {
                    'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio',
                    'description': 'Merge best video + audio',
                    'merge_output_format': 'mp4'
                },
                {
                    'format': 'worst[height>=360][ext=mp4]/worst[ext=mp4]',
                    'description': 'Lower quality MP4',
                    'merge_output_format': 'mp4'
                },
                {
                    'format': 'bestaudio[ext=m4a]/bestaudio',
                    'description': 'Audio only fallback',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'm4a',
                    }]
                },
                {
                    'format': 'best',
                    'description': 'Best available (any format)',
                    'merge_output_format': 'mp4'
                }
            ]
            
            download_success = False
            
            # Add delay before first attempt to avoid immediate bot detection
            import time
            time.sleep(2)
            
            for i, strategy in enumerate(format_strategies):
                try:
                    print(f"📥 Attempting strategy {i+1}/{len(format_strategies)}: {strategy['description']}")
                    
                    # Merge base options with strategy-specific options
                    ydl_opts = {**base_opts}
                    ydl_opts['format'] = strategy['format']
                    ydl_opts['outtmpl'] = os.path.join(temp_dir, f'video_attempt_{i}.%(ext)s')
                    
                    if 'merge_output_format' in strategy:
                        ydl_opts['merge_output_format'] = strategy['merge_output_format']
                    if 'postprocessors' in strategy:
                        ydl_opts['postprocessors'] = strategy['postprocessors']
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        # Try to extract info first
                        try:
                            print(f"  🔍 Extracting video info...")
                            info = ydl.extract_info(youtubeUrl, download=False)
                            if info:
                                print(f"  ✅ Video: '{info.get('title', 'Unknown')}' - {info.get('duration', 0)}s")
                                
                                # Check if video is age-restricted or requires auth
                                if info.get('age_limit', 0) > 0:
                                    print(f"  ⚠️ Video is age-restricted ({info.get('age_limit')}+)")
                                
                        except Exception as info_error:
                            print(f"  ⚠️ Info extraction failed: {str(info_error)[:100]}")
                            # Continue anyway - sometimes download works even when info fails
                        
                        # Attempt download
                        print(f"  ⬇️ Downloading...")
                        ydl.download([youtubeUrl])
                    
                    # Check if download was successful
                    video_files = [f for f in os.listdir(temp_dir) if f.startswith(f'video_attempt_{i}.')]
                    
                    if video_files:
                        video_file_path = os.path.join(temp_dir, video_files[0])
                        file_size = os.path.getsize(video_file_path)
                        
                        if file_size > 10240:  # At least 10KB
                            file_ext = os.path.splitext(video_files[0])[1].lower()
                            is_video_content = file_ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv']
                            
                            print(f"  ✅ Download successful!")
                            print(f"  📁 File: {video_files[0]}")
                            print(f"  📏 Size: {file_size / 1024 / 1024:.2f} MB")
                            print(f"  🎬 Type: {'Video' if is_video_content else 'Audio'}")
                            
                            download_success = True
                            break
                        else:
                            print(f"  ❌ File too small ({file_size} bytes)")
                    
                except yt_dlp.utils.DownloadError as dl_error:
                    error_msg = str(dl_error)
                    if 'Sign in to confirm' in error_msg or 'bot' in error_msg.lower():
                        print(f"  ❌ Bot detection triggered - need cookies authentication")
                    else:
                        print(f"  ❌ Download error: {error_msg[:150]}")
                    
                    # Add delay between attempts
                    if i < len(format_strategies) - 1:
                        time.sleep(3)
                    continue
                    
                except Exception as format_error:
                    print(f"  ❌ Strategy failed: {str(format_error)[:150]}")
                    if i < len(format_strategies) - 1:
                        time.sleep(2)
                    continue
            
            if not download_success:
                # Provide helpful error message
                error_msg = """
    YouTube download failed due to bot detection. Solutions:

    1. **RECOMMENDED: Use Browser Cookies**
    - Install 'Get cookies.txt LOCALLY' browser extension
    - Visit YouTube while logged in
    - Export cookies and save to your VPS
    - Update code to use: 'cookiefile': '/path/to/cookies.txt'

    2. **Alternative: Use YouTube API**
    - Get API key from Google Cloud Console
    - Use official YouTube Data API instead

    3. **Use a Proxy/VPN**
    - Route requests through residential proxy
    - Add: 'proxy': 'http://proxy:port' to ydl_opts

    4. **Try Different Video**
    - Some videos have stricter protection
    - Private/age-restricted videos won't work

    Current issue: YouTube is blocking automated downloads from your VPS IP.
    The bot detection message means you need authentication via cookies.
                """
                raise Exception(error_msg.strip())
            
            # Continue with transcription...
            print("🎤 Starting transcription...")
            try:
                import whisper
                model = whisper.load_model("base")
                result = model.transcribe(video_file_path)
                extracted_text = result["text"].strip()
                del model
                
                if extracted_text and len(extracted_text) >= 10:
                    print(f"✅ Transcription completed: {len(extracted_text)} chars")
                else:
                    raise Exception("Transcription produced no meaningful text")
                    
            except Exception as transcribe_error:
                print(f"❌ Transcription failed: {transcribe_error}")
                extracted_text = f"Audio transcription failed: {str(transcribe_error)}"
                
        except Exception as e:
            print(f"❌ YouTube processing failed: {e}")
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return {"status": "error", "message": f"YouTube processing failed: {str(e)}"}
    # Handle uploaded files with validation
    elif file:
        suffix = os.path.splitext(file.filename)[1].lower()
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
        
        try:
            # Read file content
            with os.fdopen(temp_fd, 'wb') as tmp_file:
                file_content = await file.read()
                tmp_file.write(file_content)
            
            print(f"📁 Uploaded file: {file.filename} ({len(file_content)} bytes)")
            
            # Validate uploaded file immediately
            file_validation = validate_file_for_upload(temp_path, suffix.replace('.', '').upper())
            print(f"📊 File validation results:")
            print(f"  Size: {file_validation.get('file_size_mb', 0):.2f}MB")
            print(f"  Format: {file_validation.get('format', 'Unknown')}")
            
            if not file_validation["valid"]:
                error_msg = file_validation.get('error', 'Invalid file')
                print(f"❌ Validation failed: {error_msg}")
                return {"status": "error", "message": f"File validation failed: {error_msg}"}
            
            # Show platform compatibility
            print(f"📱 Platform compatibility:")
            compatibility = file_validation.get('platform_compatibility', {})
            for platform, compat in compatibility.items():
                if platform == "tiktok":
                    if isinstance(compat, dict):
                        android_ok = compat.get('android', False)
                        ios_ok = compat.get('ios', False)
                        desktop_ok = compat.get('desktop', False)
                        limits = compat.get('limits', {})
                        current_size = compat.get('current_size', 0)
                        print(f"  TikTok ({current_size:.2f}MB):")
                        print(f"    Android (≤{limits.get('android', 0)}MB): {'✓' if android_ok else '✗'}")
                        print(f"    iOS (≤{limits.get('ios', 0)}MB): {'✓' if ios_ok else '✗'}")
                        print(f"    Desktop (≤{limits.get('desktop', 0)}MB): {'✓' if desktop_ok else '✗'}")
                else:
                    if isinstance(compat, dict):
                        is_compat = compat.get('compatible', False)
                        max_size = compat.get('max_size', 0)
                        current_size = compat.get('current_size', 0)
                        print(f"  {platform.title()} ({current_size:.2f}/{max_size}MB): {'✓' if is_compat else '✗'}")
            
            # Check if file format is supported for TikTok
            if suffix.upper().replace('.', '') not in ['MP4', 'MOV'] and any('tiktok' in str(compat) for compat in compatibility.values()):
                print(f"⚠️ Warning: {suffix.upper()} format may not be optimal for TikTok (prefers MP4/MOV)")
            
            video_file_path = temp_path
            
            if suffix in VIDEO_EXTENSIONS:
                is_video_content = True
                print(f"🎬 Detected video content: {suffix}")
            elif suffix in AUDIO_EXTENSIONS:
                print(f"🎵 Detected audio content: {suffix}")
            else:
                print(f"📄 Detected document content: {suffix}")
            
            # Extract text based on file type
            print(f"📝 Starting content extraction...")
            if suffix == ".pdf":
                extracted_text = extract_text_from_pdf(temp_path)
                print(f"✅ PDF text extracted: {len(extracted_text) if extracted_text else 0} characters")
            elif suffix == ".docx":
                extracted_text = extract_text_from_docx(temp_path)
                print(f"✅ DOCX text extracted: {len(extracted_text) if extracted_text else 0} characters")
            elif suffix == ".txt":
                with open(temp_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
                print(f"✅ TXT file read: {len(extracted_text) if extracted_text else 0} characters")
            elif suffix in AUDIO_EXTENSIONS + VIDEO_EXTENSIONS:
                try:
                    model = get_whisper_model()
                    print("🎤 Starting Whisper transcription...")
                    result = model.transcribe(temp_path)
                    extracted_text = result["text"]
                    print(f"✅ Audio/Video transcribed: {len(extracted_text) if extracted_text else 0} characters")
                except Exception as transcription_error:
                    print(f"❌ Transcription failed: {transcription_error}")
                    extracted_text = f"Transcription failed: {str(transcription_error)}"
            else:
                extracted_text = "❌ Unsupported file format."
                print(f"❌ Unsupported file format: {suffix}")
                
        except Exception as e:
            print(f"❌ File processing error: {e}")
            try:
                os.unlink(temp_path)
            except:
                pass
            return {"status": "error", "message": f"File processing failed: {str(e)}"}

    else:
        return {"status": "error", "message": "No file or YouTube URL provided"}

    # Validate extracted content
    if not extracted_text or len(extracted_text.strip()) < 10:
        print(f"❌ Insufficient content extracted: {len(extracted_text) if extracted_text else 0} characters")
        return {"status": "error", "message": "No meaningful content could be extracted from the file"}

    print(f"✅ Content extraction successful: {len(extracted_text)} characters")

    # Generate AI post
    print("🤖 Generating AI post...")
    post_content = generate_post_with_ai(extracted_text,no_of_posts)
    print(post_content)
    title = post_content
    print("in upload fiel fincop:",title)
    print(f"✅ AI post generated: {len(post_content)} characters")

    # Save to database
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO "Content"("FileType", "url", "PostContent","UserId","title")
            VALUES (%s, %s, %s,%s, %s)
            RETURNING id;
            ''',
            (fileType, source_url, post_content, user_id, title)
        )
        post_id = cursor.fetchone()[0]
        conn.commit()
        print(f"✅ Content saved to database with ID: {post_id}")
        
        clip_paths = []
        highlights = []
        
        # Process video clips if this is video content
        if post_id and is_video_content and video_file_path:
            try:
                print(f"🎬 Starting video clip generation from: {video_file_path}")
                  # Get video duration using ffprobe
                try:
                    duration_cmd = [
                        "ffprobe", "-v", "error", 
                        "-show_entries", "format=duration", 
                        "-of", "default=noprint_wrappers=1:nokey=1", 
                        video_file_path
                    ]
                    duration_result = subprocess.run(
                        duration_cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=10
                    )
                    
                    if duration_result.returncode == 0:
                        video_duration = float(duration_result.stdout.strip())
                        print(f"📹 Video duration: {video_duration:.2f} seconds ({video_duration/60:.2f} minutes)")
                        
                        # Convert to HH:MM:SS format for display
                        hours = int(video_duration // 3600)
                        minutes = int((video_duration % 3600) // 60)
                        seconds = int(video_duration % 60)
                        print(f"📹 Duration formatted: {hours:02d}:{minutes:02d}:{seconds:02d}")
                    else:
                        print(f"⚠️ Could not detect video duration, proceeding anyway")
                        video_duration = None
                        
                except Exception as duration_error:
                    print(f"⚠️ Duration detection failed: {duration_error}")
                    video_duration = None
                # Detect highlights using AI
                print("🔍 Detecting highlights with AI...")
                highlights = await detect_highlight_timestamps(extracted_text)
                print(f"✅ Found {len(highlights)} highlights:")
                for i, highlight in enumerate(highlights, 1):
                    print(f"  {i}. {highlight.get('start', 'N/A')} - {highlight.get('end', 'N/A')}: {highlight.get('reason', 'N/A')}")
                
                if highlights:
                    # Process clips with validation and embedded captions
                    print("⚡ Processing clips in parallel...")
                    start_time = time.time()
                    clip_paths = await process_clips_with_validation(
                        video_file_path, 
                        highlights, 
                        post_id, 
                        extracted_text
                    )
                    
                    end_time = time.time()
                    print(f"✅ Generated {len(clip_paths)} clips in {end_time - start_time:.1f} seconds")
                    
                    # Store clips in memory
                    if clip_paths:
                        store_clips_in_memory(post_id, clip_paths)
                        print(f"💾 Stored {len(clip_paths)} clips in memory for content {post_id}")
                        
                        # Print clip summary
                        print("\n📊 CLIP GENERATION SUMMARY:")
                        total_size = 0
                        for i, clip in enumerate(clip_paths, 1):
                            size_mb = clip.get('file_size', 0) / (1024 * 1024)
                            total_size += size_mb
                            templates_count = len(clip.get('templates', []))
                            has_captions = clip.get('has_embedded_captions', False)
                            print(f"  Clip {i}: {clip.get('title', 'N/A')} ({size_mb:.2f}MB)")
                            print(f"    Templates: {templates_count}, Captions: {'✓' if has_captions else '✗'}")
                        
                        print(f"📈 Total clips size: {total_size:.2f}MB")
                        print(f"🎯 All clips have embedded captions for better playback experience")
                    else:
                        print("⚠️ No clips were successfully generated")
                else:
                    print("⚠️ No highlights detected in the content")

            except Exception as e:
                print(f"❌ Clip generation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Clean up temporary files
        cleanup_paths = []
        if 'temp_dir' in locals():
            cleanup_paths.append(temp_dir)
        if 'temp_path' in locals():
            cleanup_paths.append(temp_path)
            
        for path in cleanup_paths:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                    print(f"🗑️ Cleaned up directory: {path}")
                elif os.path.exists(path):
                    os.unlink(path)
                    print(f"🗑️ Cleaned up file: {path}")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup warning: {cleanup_error}")
                
    except Exception as e:
        conn.rollback()
        print(f"❌ Database error: {e}")
        return {"status": "error", "message": f"Database error: {str(e)}"}
    finally:
        cursor.close()
        conn.close()

    print(f"✅ Upload processing completed successfully!")
    print(f"📊 Final stats: {len(extracted_text)} chars extracted, {len(clip_paths)} clips generated")
    
    
    # Final response
    return{
        "status": "success",
        "message": "File processed successfully",
        "extracted_text": extracted_text,
        "post_content": post_content,
        "post_id": post_id,
        "text": extracted_text,
        "highlights": highlights,
        "clips": clip_paths,
        "video_file_path": video_file_path if is_video_content else None,
        "clips_count": len(clip_paths),
        "is_video_content": is_video_content,
        "file_type": fileType
    }
    