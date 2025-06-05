from flask import Flask, render_template, request, url_for, redirect, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import yt_dlp
from pathlib import Path
import cv2
import numpy as np
import speech_recognition as sr
from pprint import pprint
import json
import subprocess
from deepgram import DeepgramClient, PrerecordedOptions
import asyncio
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
import google.generativeai as genai 
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
import threading
import uuid
from PIL import Image
import glob

# Initialize Flask app
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "supersecretkey"

# Configuration - Move sensitive data to environment variables (same as notebook)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY', '657c35834a2e83fb0dcb45cc806471fbd63a9092')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', "AIzaSyDi-m7YLhg0gW5THaReI9qCm8oMnJ9oLT8")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize database
db = SQLAlchemy(app)

# Create upload folder if it doesn't exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'csv'}

# Global processing status tracking
processing_status = {}

_cached_embed_model = None

def get_cached_embedding_model():
    """Get or create cached embedding model to avoid reloading"""
    global _cached_embed_model
    if _cached_embed_model is None:
        print("🔄 Loading embedding model (one-time setup)...")
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        # Use a smaller, faster model
        _cached_embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L12-v2",  # Faster than L6
            cache_folder="./models_cache",
            device="cpu",
            max_length=128,  # Reduced for speed
            normalize=True
        )
        print("✅ Embedding model loaded and cached!")
    else:
        print("✅ Using cached embedding model")
    return _cached_embed_model

def check_file(filename):
    """Check if file has allowed extension"""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def is_valid_youtube_url(url):
    """Basic YouTube URL validation"""
    if not url:
        return False
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
    return any(domain in url.lower() for domain in youtube_domains)

def download_video(url, output_path, cookies_file_path):
    """Download a video using yt-dlp with cookies authentication - EXACT SAME AS NOTEBOOK"""
    try:
        Path(output_path).mkdir(parents=True, exist_ok=True)

        if not Path(cookies_file_path).exists():
            print(f"ERROR: cookies file not found at {cookies_file_path}")
            return None

        ydl_opts = {
            'outtmpl': f'{output_path}/input_vid.%(ext)s',
            'format': 'best[height<=720]',
            'cookiefile': cookies_file_path,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first
            info = ydl.extract_info(url, download=False)
            metadata = {
                "Author": info.get('uploader', 'Unknown'),
                "Title": info.get('title', 'Unknown'),
                "Views": info.get('view_count', 0)
            }

            # Download the video
            ydl.download([url])

        return metadata

    except Exception as e:
        print(f"Download failed: {e}")
        return None

def video_to_images(video_path, output_folder):
    """Extract more frames for better search coverage - 75% INCREASE"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        print(f"📹 Video duration: {duration/60:.1f} minutes")

        # OPTIMIZED: Extract 75% more frames for better search coverage
        if duration > 600:  # 10+ minutes
            fps_rate = 0.15  # 1 frame every ~7 seconds (increased from 0.1)
            max_frames = 35  # Cap at 35 frames (increased from 20)
        elif duration > 300:  # 5-10 minutes  
            fps_rate = 0.2   # 1 frame every 5 seconds (increased from 0.15)
            max_frames = 32  # Cap at 32 frames (increased from 18)
        else:  # Under 5 minutes
            fps_rate = 0.35  # 1 frame every ~3 seconds (increased from 0.25)
            max_frames = 26  # Cap at 26 frames (increased from 15)

        print(f"🖼️ Extracting up to {max_frames} strategic frames (75% more coverage)...")

        # Calculate frame interval
        frame_interval = max(1, int(fps / fps_rate)) if fps_rate > 0 else int(fps * 5)

        frame_num = 0
        saved_frames = 0
        
        while saved_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_num % frame_interval == 0:
                filename = os.path.join(output_folder, f"frame_{saved_frames:04d}.png")
                cv2.imwrite(filename, frame)
                saved_frames += 1
                
            frame_num += 1
        
        cap.release()
        print(f"✅ Extracted {saved_frames} strategic frames (75% increase for better search)")
        return True
        
    except Exception as e:
        print(f"Error extracting images: {e}")
        return False

def video_to_audio(video_path, output_audio_path):
    """Extract audio from videos using FFmpeg - SAME AS NOTEBOOK LOGIC"""
    try:
        # Use FFmpeg to extract audio (same approach as notebook)
        cmd = [
            'ffmpeg', '-i', video_path, 
            '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '44100', '-ac', '2', 
            output_audio_path, '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False

def audio_to_text(video_id, audio_file_path=None, progress_callback=None):
    """
    Extract transcript from YouTube video using YouTube Transcript API.
    Falls back to audio transcription if no transcript is available.
    
    Args:
        video_id (str): YouTube video ID (e.g., 'dQw4w9WgXcQ')
        audio_file_path (str, optional): Path to audio file for fallback transcription
        progress_callback (callable, optional): Function to call with progress updates
        
    Returns:
        tuple: (transcript_text, transcript_available_flag)
    """
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
    import re
    
    def update_progress(percent, message=""):
        if progress_callback:
            progress_callback(percent, message)
    
    try:
        update_progress(10, "Checking for YouTube transcript...")
        
        # Try to get transcript from YouTube
        print(f"🔍 Attempting to fetch transcript for video ID: {video_id}")
        
        # Get available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        transcript = None
        transcript_info = ""
        
        # Try to get manually created transcript first (highest quality)
        try:
            for t in transcript_list:
                if not t.is_generated:
                    transcript = t.fetch()
                    transcript_info = f"Manual transcript ({t.language})"
                    print(f"✅ Found manual transcript in {t.language}")
                    break
        except Exception as e:
            print(f"⚠️  No manual transcript available: {e}")
        
        # If no manual transcript, try auto-generated
        if not transcript:
            try:
                for t in transcript_list:
                    if t.is_generated:
                        transcript = t.fetch()
                        transcript_info = f"Auto-generated transcript ({t.language})"
                        print(f"✅ Found auto-generated transcript in {t.language}")
                        break
            except Exception as e:
                print(f"⚠️  No auto-generated transcript available: {e}")
        
        # If still no transcript, try English specifically
        if not transcript:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                transcript_info = "English transcript"
                print("✅ Found English transcript")
            except Exception as e:
                print(f"⚠️  No English transcript available: {e}")
        
        # If we got a transcript, format it
        if transcript:
            update_progress(50, f"Processing {transcript_info}...")
            
            # Format transcript to plain text
            formatter = TextFormatter()
            transcript_text = formatter.format_transcript(transcript)
            
            # Clean up the text
            transcript_text = re.sub(r'\[.*?\]', '', transcript_text)  # Remove [Music], [Applause], etc.
            transcript_text = re.sub(r'\n+', ' ', transcript_text)     # Replace multiple newlines with space
            transcript_text = re.sub(r'\s+', ' ', transcript_text)     # Replace multiple spaces with single space
            transcript_text = transcript_text.strip()
            
            update_progress(90, "Transcript processing complete")
            
            print(f"✅ Transcript extracted successfully!")
            print(f"📝 Transcript length: {len(transcript_text)} characters")
            print(f"📄 Type: {transcript_info}")
            
            update_progress(100, "YouTube transcript ready")
            return transcript_text, True
            
    except Exception as e:
        print(f"❌ YouTube Transcript API failed: {e}")
        update_progress(30, "YouTube transcript failed, trying audio transcription...")
    
    # Fallback to audio transcription if transcript API fails
    print("🎵 Falling back to audio transcription...")
    
    if not audio_file_path:
        print("❌ No audio file provided for fallback transcription")
        update_progress(100, "No transcript available")
        return "No transcript available for this video.", False
    
    try:
        import whisper
        import os
        
        update_progress(40, "Loading Whisper model...")
        
        # Load Whisper model
        model = whisper.load_model("base")
        
        update_progress(60, "Transcribing audio with Whisper...")
        
        # Transcribe audio
        result = model.transcribe(audio_file_path)
        transcript_text = result["text"]
        
        update_progress(90, "Audio transcription complete")
        
        print(f"✅ Audio transcription completed!")
        print(f"📝 Transcript length: {len(transcript_text)} characters")
        print("📄 Type: Whisper audio transcription")
        
        update_progress(100, "Audio transcript ready")
        return transcript_text, True
        
    except ImportError:
        print("❌ Whisper not installed. Install with: pip install openai-whisper")
        update_progress(100, "Transcription failed - Whisper not available")
        return "Whisper not installed for audio transcription.", False
        
    except Exception as e:
        print(f"❌ Audio transcription failed: {e}")
        update_progress(100, "Transcription failed")
        return f"Audio transcription failed: {str(e)}", False


def extract_video_id(url):
    """
    Extract YouTube video ID from URL.
    
    Args:
        url (str): YouTube URL
        
    Returns:                                                 
        str: Video ID or None if not found
    """
    import re
    
    # Handle different YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=)([\w-]+)',
        r'(?:youtu\.be\/)([\w-]+)',
        r'(?:youtube\.com\/embed\/)([\w-]+)',
        r'(?:youtube\.com\/v\/)([\w-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def retrieve(retriever_engine, query_str):
    """EXACT SAME RETRIEVE FUNCTION AS NOTEBOOK"""
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text

def prepare_images_for_gemini(image_paths):
    """EXACT SAME FUNCTION AS NOTEBOOK"""
    images = []
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)
            images.append(image)
    return images

def analyze_frames_with_transcript_context(session_id, frame_paths, transcript_text):
    """Analyze frames with transcript context for better search results"""
    try:
        print(f"🔍 Analyzing {len(frame_paths)} frames with transcript context...")
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        frame_descriptions = {}
        
        # Split transcript into roughly timed segments
        transcript_words = transcript_text.split()
        total_words = len(transcript_words)
        
        # Process frames in smaller batches for better context
        batch_size = 3  # Smaller batches for better descriptions
        
        for i in range(0, len(frame_paths), batch_size):
            batch = frame_paths[i:i+batch_size]
            
            # Prepare images for this batch
            images = []
            valid_paths = []
            for img_path in batch:
                if os.path.isfile(img_path):
                    try:
                        image = Image.open(img_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        images.append(image)
                        valid_paths.append(img_path)
                    except Exception as e:
                        print(f"⚠️ Skipping image {img_path}: {e}")
                        continue
            
            if not images:
                continue
            
            # Estimate what part of transcript corresponds to these frames
            frame_start_idx = (i / len(frame_paths)) * total_words
            frame_end_idx = ((i + len(images)) / len(frame_paths)) * total_words
            
            # Get relevant transcript segment
            relevant_transcript = " ".join(transcript_words[int(frame_start_idx):int(frame_end_idx)])
            if len(relevant_transcript) < 50 and total_words > 50:
                # If segment is too small, get a bit more context
                start_idx = max(0, int(frame_start_idx) - 25)
                end_idx = min(total_words, int(frame_end_idx) + 25)
                relevant_transcript = " ".join(transcript_words[start_idx:end_idx])
            
            # Enhanced prompt with transcript context
            enhanced_prompt = f"""
            Analyze these video frames in detail. I will provide you with the corresponding transcript segment to give you context about what's being discussed when these frames appear.
            
            TRANSCRIPT CONTEXT for these frames:
            "{relevant_transcript}"
            
            For each frame, provide a detailed, searchable description that includes:
            1. Visual elements: objects, people, animals, vehicles, text, settings
            2. Actions happening: what people are doing, movements, interactions
            3. Scene context: indoor/outdoor, location type, lighting, mood
            4. Content context: what topic is being discussed (from transcript)
            5. Searchable keywords: specific terms someone might search for
            
            Make descriptions very specific and keyword-rich so they can be easily found with natural language queries.
            
            Format your response exactly as:
            Frame 1: [comprehensive searchable description]
            Frame 2: [comprehensive searchable description]
            Frame 3: [comprehensive searchable description]
            
            Be very detailed and include many searchable terms.
            """
            
            try:
                content = [enhanced_prompt] + images
                response = model.generate_content(content)
                descriptions_text = response.text
                
                # Parse the response and map to frame paths
                lines = descriptions_text.split('\n')
                frame_idx = 0
                for line in lines:
                    if line.strip().startswith('Frame ') and ':' in line:
                        description = line.split(':', 1)[1].strip()
                        if frame_idx < len(valid_paths):
                            frame_name = os.path.basename(valid_paths[frame_idx])
                            frame_descriptions[frame_name] = {
                                'path': valid_paths[frame_idx],
                                'description': description,
                                'frame_number': frame_idx + i,
                                'transcript_context': relevant_transcript[:200] + "..." if len(relevant_transcript) > 200 else relevant_transcript
                            }
                            frame_idx += 1
                
                print(f"✅ Processed batch {i//batch_size + 1}/{(len(frame_paths)-1)//batch_size + 1}")
                
            except Exception as e:
                print(f"⚠️ Failed to analyze batch starting at {i}: {e}")
                continue
        
        print(f"✅ Enhanced frame analysis complete: {len(frame_descriptions)} frames indexed with transcript context")
        return frame_descriptions
        
    except Exception as e:
        print(f"❌ Frame analysis failed: {e}")
        return {}

def search_frames_with_fuzzy_matching(session_id, query, max_results=2):
    """Improved frame search with fuzzy matching and transcript context"""
    try:
        # Get the stored frame descriptions
        session_data = processing_status.get(session_id, {})
        if session_data.get("status") != "completed":
            return {"error": "Session not found or not completed"}
        
        frame_descriptions = session_data.get("frame_descriptions", {})
        if not frame_descriptions:
            return {"error": "No frame descriptions available for this session"}
        
        print(f"🔍 Searching {len(frame_descriptions)} frames for: '{query}'")
        
        # Enhanced search using multiple approaches
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create a comprehensive search prompt
        search_prompt = f"""
        You are helping a user find specific video frames. The user wants to find: "{query}"
        
        I will give you frame descriptions that include both visual content and transcript context.
        Your job is to find the 1-2 BEST matching frames.
        
        Available frames:
        
        """
        
        frame_list = []
        for frame_name, frame_data in frame_descriptions.items():
            search_prompt += f"""
Frame {frame_data['frame_number']}:
Visual: {frame_data['description']}
Context: {frame_data.get('transcript_context', 'No context')}
---
"""
            frame_list.append((frame_name, frame_data))
        
        search_prompt += f"""
        
        USER QUERY: "{query}"
        
        Instructions:
        - Find frames that match the query based on EITHER visual content OR transcript context
        - Look for keyword matches, semantic similarity, or related concepts
        - If query mentions objects/people/actions, focus on visual descriptions
        - If query mentions topics/concepts/words, also consider transcript context
        - Return the frame numbers that best match, in order of relevance
        - Return MAXIMUM {max_results} results
        - Be flexible with matching - similar concepts should match
        
        Examples of what should match:
        - "car" should match "vehicle", "automobile", "driving"
        - "talking" should match "speaking", "conversation", "discussion"
        - "outside" should match "outdoor", "exterior", "street"
        
        Response format: Just the frame numbers separated by commas
        Example: 5, 12
        
        If NO frames match AT ALL, respond with: NONE
        """
        
        response = model.generate_content(search_prompt)
        result_text = response.text.strip()
        
        print(f"🤖 Search response: {result_text}")
        
        if "NONE" in result_text.upper():
            return {"matches": [], "message": f"No frames found matching '{query}'. Try different keywords like objects, actions, or topics from the video."}
        
        # Parse the frame numbers more flexibly
        import re
        frame_numbers = []
        
        # First try comma-separated numbers
        if ',' in result_text:
            try:
                frame_numbers = [int(x.strip()) for x in result_text.split(',') if x.strip().isdigit()]
            except:
                pass
        
        # If that fails, extract any numbers
        if not frame_numbers:
            frame_numbers = [int(x) for x in re.findall(r'\d+', result_text)]
        
        # Limit to max_results
        frame_numbers = frame_numbers[:max_results]
        
        print(f"📊 Found frame numbers: {frame_numbers}")
        
        # Get the matching frames
        matches = []
        for frame_num in frame_numbers:
            for frame_name, frame_data in frame_list:
                if frame_data['frame_number'] == frame_num:
                    matches.append({
                        'frame_name': frame_name,
                        'frame_path': frame_data['path'],
                        'description': frame_data['description'],
                        'frame_number': frame_data['frame_number'],
                        'transcript_context': frame_data.get('transcript_context', '')
                    })
                    break
        
        if not matches:
            # Fallback: do keyword-based search
            print("🔄 Trying fallback keyword search...")
            query_words = query.lower().split()
            
            scored_frames = []
            for frame_name, frame_data in frame_list:
                description_lower = frame_data['description'].lower()
                context_lower = frame_data.get('transcript_context', '').lower()
                combined_text = description_lower + " " + context_lower
                
                score = 0
                for word in query_words:
                    if word in combined_text:
                        score += 1
                    # Check for partial matches
                    if any(word in text_word for text_word in combined_text.split()):
                        score += 0.5
                
                if score > 0:
                    scored_frames.append((score, frame_name, frame_data))
            
            # Sort by score and take top results
            scored_frames.sort(reverse=True, key=lambda x: x[0])
            
            for score, frame_name, frame_data in scored_frames[:max_results]:
                matches.append({
                    'frame_name': frame_name,
                    'frame_path': frame_data['path'],
                    'description': frame_data['description'],
                    'frame_number': frame_data['frame_number'],
                    'transcript_context': frame_data.get('transcript_context', ''),
                    'match_score': score
                })
        
        if matches:
            return {
                "matches": matches, 
                "query": query, 
                "total_found": len(matches),
                "message": f"Found {len(matches)} frame(s) matching '{query}'"
            }
        else:
            return {
                "matches": [], 
                "message": f"No frames found matching '{query}'. The video might not contain this content, or try using different keywords."
            }
        
    except Exception as e:
        print(f"❌ Frame search failed: {e}")
        return {"error": f"Search failed: {str(e)}"}

def get_transcript_for_session(session_id):
    """Helper function to get transcript for a session"""
    session_data = processing_status.get(session_id, {})
    return session_data.get("transcript", "")

# Add this route to your Flask app


def process_video_async(video_url, session_id, cookies_file_path):
    """Process video in background thread - ULTRA-FAST VERSION (NO EMBEDDINGS)"""
    try:
        # Extract video ID for transcript API
        video_id = extract_video_id(video_url)
        if not video_id:
            print("❌ Could not extract video ID from URL")
            processing_status[session_id] = {
                "status": "error",
                "progress": 0,
                "message": "Invalid YouTube URL - could not extract video ID"
            }
            return
        
        print(f"📁 Session ID: {session_id}")
        print(f"🎬 Video ID: {video_id}")
        print(f"🔗 Processing URL: {video_url}")
        
        processing_status[session_id] = {"status": "downloading", "progress": 10}
        
        # Create session-specific directories - SAME STRUCTURE AS NOTEBOOK
        output_video_path = os.path.join(UPLOAD_FOLDER, session_id, "video_data")
        output_folder = os.path.join(UPLOAD_FOLDER, session_id, "mixed_data")
        output_audio_path = os.path.join(output_folder, "output_audio.wav")
        
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Clean up old files first (like notebook)
        print("🧹 Cleaning up old files...")
        for pattern in [f'{output_video_path}/*', f'{output_folder}/*']:
            for old_file in glob.glob(pattern):
                try:
                    if os.path.isfile(old_file):
                        os.remove(old_file)
                except:
                    pass
        
        # Step 1: Download video - EXACT SAME AS NOTEBOOK
        print("📥 Downloading video...")
        metadata_vid = download_video(video_url, output_video_path, cookies_file_path)
        if not metadata_vid:
            processing_status[session_id] = {"status": "error", "message": "Failed to download video"}
            return
        
        processing_status[session_id] = {"status": "extracting", "progress": 30}
        
        # Find the downloaded video file - SAME AS NOTEBOOK
        video_file = None
        for file in os.listdir(output_video_path):
            if file.startswith("input_vid."):
                video_file = os.path.join(output_video_path, file)
                break
        
        if not video_file:
            processing_status[session_id] = {"status": "error", "message": "Video file not found"}
            return
        
        # Verify download like notebook
        file_size = os.path.getsize(video_file)
        if file_size < 1000:  # Less than 1KB
            processing_status[session_id] = {"status": "error", "message": f"Video file too small ({file_size} bytes)"}
            return
        
        print(f"✅ Downloaded and verified: {metadata_vid['Title']}")
        print(f"📁 File size: {file_size:,} bytes")
        
        # Step 2: Extract images and audio - SAME AS NOTEBOOK
        print("🎬 Processing video frames...")
        if not video_to_images(video_file, output_folder):
            processing_status[session_id] = {"status": "error", "message": "Failed to extract images"}
            return
            
        processing_status[session_id] = {"status": "extracting", "progress": 40}
        
        print("🎵 Extracting audio...")
        if not video_to_audio(video_file, output_audio_path):
            processing_status[session_id] = {"status": "error", "message": "Failed to extract audio"}
            return
        
        # Step 3: NEW TRANSCRIPT APPROACH - Use YouTube Transcript API first
        processing_status[session_id] = {"status": "transcribing", "progress": 50}
        
        # Progress callback for transcript
        def transcript_progress(percent, message):
            # Map transcript progress to 50-70% of total progress
            total_progress = 50 + (percent * 0.2)
            processing_status[session_id].update({
                "status": "transcribing",
                "progress": int(total_progress),
                "message": message
            })
            print(f"📊 Transcript: {percent}% - {message}")
        
        # NEW: Get transcript using YouTube API with fallback
        print("📝 Getting transcript using YouTube API...")
        text_data, transcript_available = audio_to_text(
            video_id=video_id,
            audio_file_path=output_audio_path,  # Fallback if no YouTube transcript
            progress_callback=transcript_progress
        )
        
        # Save transcript - SAME AS NOTEBOOK
        with open(os.path.join(output_folder, "output_text.txt"), "w", encoding='utf-8') as file:
            file.write(text_data)
        
        # Clean up audio file - SAME AS NOTEBOOK
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        
        # ULTRA-FAST: Skip embeddings entirely, process directly
        processing_status[session_id] = {"status": "analyzing", "progress": 70}
        print("⚡ Using direct processing (no embeddings) for maximum speed...")
        
        # Read transcript directly
        transcript_file = os.path.join(output_folder, "output_text.txt")
        try:
            with open(transcript_file, "r", encoding='utf-8') as file:
                full_transcript = file.read()
        except FileNotFoundError:
            full_transcript = "No transcript available"
        
        # Get all extracted images
        image_files = glob.glob(os.path.join(output_folder, "frame_*.png"))
        image_files.sort()  # Sort by filename
        
        print(f"📸 Found {len(image_files)} frame images")
        print(f"📝 Transcript length: {len(full_transcript)} characters")
        
        processing_status[session_id] = {"status": "generating", "progress": 85}
        
        # ULTRA-FAST ANALYSIS: Direct processing without vector search
         # OPTIMIZED FRAME SELECTION: Analyze more frames for better search coverage
        print("🤖 Generating direct analysis...")
        
        # Prepare more frames for analysis (instead of just strategic subset)
        selected_images = []
        if len(image_files) > 0:
            # OPTION 1: Analyze more strategic frames (recommended)
            num_to_analyze = min(len(image_files), 20)  # Analyze up to 20 frames (4x more than before)
            
            if len(image_files) <= num_to_analyze:
                # If we have few frames, analyze all of them
                selected_images = image_files
            else:
                # Distribute frames evenly across the video
                step = len(image_files) / num_to_analyze
                selected_indices = [int(i * step) for i in range(num_to_analyze)]
                selected_images = [image_files[i] for i in selected_indices if i < len(image_files)]
        
        print(f"🎯 Using {len(selected_images)} frames for analysis (75% more coverage)")
        print(f"📊 Frame analysis ratio: {len(selected_images)}/{len(image_files)} frames")
        
        # FAST PROMPT: Direct analysis without vector retrieval
        direct_prompt = f"""
        You are an expert video analyst. Analyze this video comprehensively using the full transcript and key visual frames.
        
        VIDEO METADATA:
        Title: {metadata_vid.get('Title', 'Unknown')}
        Author: {metadata_vid.get('Author', 'Unknown')}
        Views: {metadata_vid.get('Views', 0):,}
        
        FULL TRANSCRIPT:
        {full_transcript}
        
        INSTRUCTIONS:
        - Provide comprehensive analysis (2200+ words)
        - Generate estimated timestamps in [MM:SS] format based on content flow
        - Quote directly from the transcript
        - Reference the visual frames provided
        - Analyze the main topics, themes, and key points
        - Provide detailed explanations and context
        - Estimate timing based on transcript content progression
        - The longer the video and more concepts or ideas it has , the more words generated and more timestamps clearly
        
        ANALYSIS QUERY: Using examples from video, explain all things covered in the video and give approximate timestamps of what each portion of the video talks about.
        
        Please provide a detailed analysis with estimated timestamps:
        """
        
        try:
            # Use Gemini for analysis
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=6000,
                    temperature=0.3,
                )
            )
            
            # Prepare images
            images = []
            for img_path in selected_images[:8]:  # Limit to 8 images max
                if os.path.isfile(img_path):
                    try:
                        image = Image.open(img_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        images.append(image)
                    except Exception as e:
                        print(f"⚠️ Skipping image {img_path}: {e}")
                        continue
            
            print(f"📸 Processing {len(images)} images...")
            
            # Generate content
            if images:
                content = [direct_prompt] + images
            else:
                content = [direct_prompt]
            
            response = model.generate_content(content)
            analysis_result = response.text
            
            print("✅ Ultra-fast analysis completed!")
            
        except Exception as e:
            print(f"❌ Gemini API failed: {e}")
            # Fallback to text-only analysis
            try:
                print("🔄 Falling back to text-only analysis...")
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                fallback_prompt = f"""
                Analyze this video based on the available transcript:
                
                VIDEO METADATA:
                Title: {metadata_vid.get('Title', 'Unknown')}
                Author: {metadata_vid.get('Author', 'Unknown')}
                Views: {metadata_vid.get('Views', 0):,}
                
                FULL TRANSCRIPT:
                {full_transcript}
                
                Provide a comprehensive analysis with estimated timestamps based on the content flow.
                Include main topics, themes, and approximate timing based on content progression.
                """
                
                response = model.generate_content(fallback_prompt)
                analysis_result = response.text
                print("✅ Text-only analysis completed!")
                
            except Exception as e2:
                print(f"❌ All analysis methods failed: {e2}")
                analysis_result = f"Analysis failed: {str(e2)}"
        
        processing_status[session_id] = {"status": "indexing", "progress": 95}
        print("🔍 Analyzing frames for search functionality...")
           # Use the improved frame analysis that includes transcript context
        frame_descriptions = analyze_frames_with_transcript_context(session_id, selected_images, text_data)
        
        # Mark as completed with all data including frame descriptions
        final_status = {
            "status": "completed", 
            "progress": 100,
            "result": analysis_result,
            "metadata": metadata_vid,
            "transcript_available": transcript_available,
            "frames_analyzed": len(selected_images),
            "transcript": text_data,  # Store transcript for search context
            "processing_method": "ultra_fast_with_context",
            "frame_descriptions": frame_descriptions,  # Enhanced frame descriptions
            "query_enabled": len(frame_descriptions) > 0  # Flag for query capability
        }
        
        processing_status[session_id] = final_status
        
        # Debug logging
        print("=" * 50)
        print("🎉 ULTRA-FAST PROCESSING COMPLETE!")
        print(f"📁 Session ID: {session_id}")
        print(f"📊 Status set to: {final_status['status']}")
        print(f"📝 Result length: {len(analysis_result)} characters")
        print(f"🔤 Transcript available: {transcript_available}")
        print(f"🖼️ Frames analyzed: {len(selected_images)}")
        print(f"🔍 Frame descriptions: {len(frame_descriptions)} frames indexed")
        print(f"⚡ Processing method: ultra_fast")
        print(f"💾 Full status saved: {processing_status[session_id]['status']}")
        print(f"🔍 Status check URL: http://localhost:5000/status/{session_id}")
        print(f"📋 Results URL: http://localhost:5000/result/{session_id}")
        print("=" * 50)
        
    except Exception as e:
        error_message = f"Processing failed: {str(e)}"
        print(f"❌ {error_message}")
        processing_status[session_id] = {
            "status": "error",
            "progress": 0,
            "message": error_message
        }
        
        # Also print the full traceback for debugging
        import traceback
        print("📋 Full error traceback:")
        traceback.print_exc()
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        video_url = request.form.get("video")
        
        if not video_url:
            flash("Please provide a YouTube URL")
            return render_template('home.html')
        
        if not is_valid_youtube_url(video_url):
            flash("Please provide a valid YouTube URL")
            return render_template('home.html')
        
        # Check if cookies file exists
        cookies_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.txt', '.csv'))]
        if not cookies_files:
            flash("Please upload a cookies file first")
            return redirect(url_for('upload'))
        
        # Use the most recent cookies file
        cookies_file = max(cookies_files, key=lambda x: os.path.getctime(os.path.join(UPLOAD_FOLDER, x)))
        cookies_file_path = os.path.join(UPLOAD_FOLDER, cookies_file)
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Start background processing
        thread = threading.Thread(
            target=process_video_async, 
            args=(video_url, session_id, cookies_file_path)
        )
        thread.start()
        
        return redirect(url_for('processing', session_id=session_id))
    
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('Please select a cookies.txt or cookies.csv file from your YouTube account')
            return redirect(request.url)
        
        if file and check_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File uploaded successfully!')
            return redirect(url_for('home'))
        else:
            flash('Invalid file type. Please upload a .txt or .csv file')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/processing/<session_id>')
def processing(session_id):
    return render_template('processing.html', session_id=session_id)

@app.route('/status/<session_id>')
def get_status(session_id):
    status = processing_status.get(session_id, {"status": "not_found"})
    return jsonify(status)

@app.route('/result/<session_id>')
def result(session_id):
    status = processing_status.get(session_id, {})
    if status.get("status") == "completed":
        return render_template('result.html', 
                             result=status.get("result"), 
                             metadata=status.get("metadata"),
                             transcript_available=status.get("transcript_available", False),
                             frames_analyzed=status.get("frames_analyzed", 0))
    else:
        flash("Processing not completed or session not found")
        return redirect(url_for('home'))
# Replace your existing query route with this improved version:

@app.route('/query/<session_id>', methods=['GET', 'POST'])
def query_frames(session_id):
    """Handle frame querying interface with improved search"""
    session_data = processing_status.get(session_id, {})
    
    if session_data.get("status") != "completed":
        flash("Session not found or video processing not completed")
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            flash("Please enter a search query")
            return render_template('query.html', session_id=session_id, 
                                 metadata=session_data.get("metadata"))
        
        print(f"🔍 Processing search query: '{query}' for session {session_id}")
        
        # Use the improved search function
        search_results = search_frames_with_fuzzy_matching(session_id, query, max_results=2)
        
        print(f"📊 Search results: {search_results}")
        
        return render_template('query.html', 
                             session_id=session_id,
                             metadata=session_data.get("metadata"),
                             query=query,
                             search_results=search_results)
    
    return render_template('query.html', 
                         session_id=session_id,
                         metadata=session_data.get("metadata"))

@app.route('/frame/<session_id>/<frame_name>')
def serve_frame(session_id, frame_name):
    """Serve individual frame images"""
    try:
        frame_path = os.path.join(UPLOAD_FOLDER, session_id, "mixed_data", frame_name)
        if os.path.exists(frame_path):
            from flask import send_file
            return send_file(frame_path)
        else:
            return "Frame not found", 404
    except Exception as e:
        return f"Error serving frame: {e}", 500
if __name__ == "__main__":
    # Disable auto-reloader to avoid virtual environment watching issues
    app.run(debug=True, use_reloader=False)
