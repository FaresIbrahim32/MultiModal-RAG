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
    """Extract fewer, strategic frames for faster processing"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        print(f"📹 Video duration: {duration/60:.1f} minutes")

        # OPTIMIZED: Extract fewer frames strategically
        if duration > 600:  # 10+ minutes
            fps_rate = 0.1  # 1 frame every 10 seconds
            max_frames = 20  # Cap at 20 frames
        elif duration > 300:  # 5-10 minutes  
            fps_rate = 0.15  # 1 frame every ~7 seconds
            max_frames = 18
        else:  # Under 5 minutes
            fps_rate = 0.25  # 1 frame every 4 seconds
            max_frames = 15

        print(f"🖼️ Extracting up to {max_frames} strategic frames...")

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
        print(f"✅ Extracted {saved_frames} strategic frames")
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

def process_video_async(video_url, session_id, cookies_file_path):
    """Process video with OPTIMIZED LlamaIndex embeddings - Fast + Accurate"""
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
        
        # Create session-specific directories
        output_video_path = os.path.join(UPLOAD_FOLDER, session_id, "video_data")
        output_folder = os.path.join(UPLOAD_FOLDER, session_id, "mixed_data")
        output_audio_path = os.path.join(output_folder, "output_audio.wav")
        
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Clean up old files
        print("🧹 Cleaning up old files...")
        for pattern in [f'{output_video_path}/*', f'{output_folder}/*']:
            for old_file in glob.glob(pattern):
                try:
                    if os.path.isfile(old_file):
                        os.remove(old_file)
                except:
                    pass
        
        # Step 1: Download video
        print("📥 Downloading video...")
        metadata_vid = download_video(video_url, output_video_path, cookies_file_path)
        if not metadata_vid:
            processing_status[session_id] = {"status": "error", "message": "Failed to download video"}
            return
        
        processing_status[session_id] = {"status": "extracting", "progress": 30}
        
        # Find the downloaded video file
        video_file = None
        for file in os.listdir(output_video_path):
            if file.startswith("input_vid."):
                video_file = os.path.join(output_video_path, file)
                break
        
        if not video_file:
            processing_status[session_id] = {"status": "error", "message": "Video file not found"}
            return
        
        # Verify download
        file_size = os.path.getsize(video_file)
        if file_size < 1000:
            processing_status[session_id] = {"status": "error", "message": f"Video file too small ({file_size} bytes)"}
            return
        
        print(f"✅ Downloaded: {metadata_vid['Title']}")
        print(f"📁 File size: {file_size:,} bytes")
        
        # Step 2: Extract images (OPTIMIZED)
        print("🎬 Processing video frames (optimized)...")
        if not video_to_images(video_file, output_folder):  # Use optimized version
            processing_status[session_id] = {"status": "error", "message": "Failed to extract images"}
            return
            
        processing_status[session_id] = {"status": "extracting", "progress": 40}
        
        print("🎵 Extracting audio...")
        if not video_to_audio(video_file, output_audio_path):
            processing_status[session_id] = {"status": "error", "message": "Failed to extract audio"}
            return
        
        # Step 3: Get transcript
        processing_status[session_id] = {"status": "transcribing", "progress": 50}
        
        def transcript_progress(percent, message):
            total_progress = 50 + (percent * 0.15)  # Reduced time allocation
            processing_status[session_id].update({
                "status": "transcribing",
                "progress": int(total_progress),
                "message": message
            })
            print(f"📊 Transcript: {percent}% - {message}")
        
        print("📝 Getting transcript using YouTube API...")
        text_data, transcript_available = audio_to_text(
            video_id=video_id,
            audio_file_path=output_audio_path,
            progress_callback=transcript_progress
        )
        
        # Save transcript
        with open(os.path.join(output_folder, "output_text.txt"), "w", encoding='utf-8') as file:
            file.write(text_data)
        
        # Clean up audio file
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        
        # Step 4: OPTIMIZED EMBEDDINGS (The key improvement!)
        processing_status[session_id] = {"status": "analyzing", "progress": 65}
        
        # SIMPLIFIED: Use default vector stores (no custom imports needed)
        print("🔍 Building optimized search index...")
        
        # Use cached embedding model
        Settings.embed_model = get_cached_embedding_model()
        
        # Load documents
        documents = SimpleDirectoryReader(output_folder).load_data()
        print(f"📄 Processing {len(documents)} documents...")
        
        # Build index with default settings (much simpler and faster)
        index = MultiModalVectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
        
        # OPTIMIZED: Reduce retrieval parameters for speed
        retriever_engine = index.as_retriever(
            similarity_top_k=3,      # Reduced from 5
            image_similarity_top_k=10 # Reduced from 15
        )
        
        processing_status[session_id] = {"status": "generating", "progress": 85}
        
        # Step 5: Perform analysis
        query_str = "Using examples from video, explain all things covered in the video and give approximate timestamps of what each portion of the video talks about"
        
        print("🎯 Retrieving relevant content...")
        img, txt = retrieve(retriever_engine=retriever_engine, query_str=query_str)
        
        # Enhanced context with transcript
        context_str = "".join(txt)
        
        try:
            with open(os.path.join(output_folder, "output_text.txt"), "r", encoding='utf-8') as file:
                google_transcript = file.read()

            if google_transcript.strip():
                enhanced_context = context_str + f"\n\n=== FULL VIDEO TRANSCRIPT ===\n{google_transcript}\n"
                print(f"✅ Added transcript ({len(google_transcript)} characters)")
            else:
                enhanced_context = context_str

        except FileNotFoundError:
            print("⚠️ No transcript file found")
            enhanced_context = context_str
        
        # Generate analysis
        enhanced_qa_tmpl = (
            "You are an expert video analyst. Analyze this video comprehensively using the visual frames and transcript. "
            "Since exact timestamps aren't available, estimate approximate timestamps based on video segments and frame sequence. "
            "IMPORTANT: Generate estimated timestamps in [MM:SS] format based on video length and content flow.\n\n"
            "INSTRUCTIONS:\n"
            "- Provide comprehensive analysis (1200+ words)\n"
            "- Generate estimated timestamps like [02:15] based on content progression\n"
            "- Quote from the transcript and reference specific visual elements\n"
            "- Analyze frame sequence to estimate timing\n"
            "- Provide detailed explanations and context\n\n"
            "Available Context: {context_str}\n"
            "Video Metadata: {metadata_str}\n"
            "Query: {query_str}\n\n"
            "Comprehensive Analysis with Estimated Timestamps:"
        )
        
        metadata_str = json.dumps(metadata_vid)
        
        print("🤖 Generating AI analysis...")
        
        try:
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=6000,
                    temperature=0.3,
                )
            )

            full_prompt = enhanced_qa_tmpl.format(
                context_str=enhanced_context,
                query_str=query_str,
                metadata_str=metadata_str
            )

            # Prepare images (limit for speed)
            images = []
            for img_path in img[:6]:  # Reduced to 6 images
                if os.path.isfile(img_path):
                    try:
                        image = Image.open(img_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        images.append(image)
                    except Exception as e:
                        print(f"⚠️ Skipping image {img_path}: {e}")
                        continue

            print(f"📸 Processing {len(images)} relevant images...")

            # Generate content
            if images:
                content = [full_prompt] + images
            else:
                content = [full_prompt]

            response_1 = model.generate_content(content)
            analysis_result = response_1.text
            
            print("✅ Analysis generated successfully!")
            
        except Exception as e:
            print(f"❌ Gemini API failed: {e}")
            # Fallback to text-only
            try:
                print("🔄 Falling back to text-only analysis...")
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                fallback_prompt = f"""
                Analyze this video based on the retrieved context and transcript:
                
                {enhanced_context}
                
                Video Metadata: {metadata_str}
                
                Query: {query_str}
                
                Provide a comprehensive analysis with estimated timestamps based on the content flow.
                """
                
                response_1 = model.generate_content(fallback_prompt)
                analysis_result = response_1.text
                print("✅ Text-only analysis completed!")
                
            except Exception as e2:
                print(f"❌ All analysis methods failed: {e2}")
                analysis_result = f"Analysis failed: {str(e2)}"
        
        # Mark as completed
        final_status = {
            "status": "completed", 
            "progress": 100,
            "result": analysis_result,
            "metadata": metadata_vid,
            "transcript_available": transcript_available,
            "frames_analyzed": len(img),
            "transcript": text_data,
            "processing_method": "optimized_embeddings"
        }
        
        processing_status[session_id] = final_status
        
        print("=" * 50)
        print("🎉 OPTIMIZED PROCESSING COMPLETE!")
        print(f"📁 Session ID: {session_id}")
        print(f"📊 Status: {final_status['status']}")
        print(f"📝 Result length: {len(analysis_result)} characters")
        print(f"🔤 Transcript available: {transcript_available}")
        print(f"🖼️ Relevant frames found: {len(img)}")
        print(f"⚡ Method: optimized_embeddings")
        print(f"🔍 Status URL: http://localhost:5000/status/{session_id}")
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

if __name__ == "__main__":
    # Disable auto-reloader to avoid virtual environment watching issues
    app.run(debug=True, use_reloader=False)
