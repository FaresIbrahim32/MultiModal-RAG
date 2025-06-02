# -*- coding: utf-8 -*-
import os
import json
import time
from pathlib import Path
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import yt_dlp
from moviepy.editor import VideoFileClip
from deepgram import DeepgramClient, PrerecordedOptions
import google.generativeai as genai
import speech_recognition as sr

# Import LlamaIndex components
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.multi_modal_llms.gemini import GeminiMultiModal

# Configuration
output_video_path = "./video_data/"
output_folder = "./mixed_data/"
output_audio_path = "./mixed_data/output_audio.wav"
filepath = output_video_path + "input_vid.mp4"

# Initialize settings
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# Initialize vector stores
text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)

# Ensure directories exist
Path(output_folder).mkdir(parents=True, exist_ok=True)
Path(output_video_path).mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---
def download_video(url, output_path):
    """Download a video using yt-dlp with cookies authentication."""
    if not Path('./cookies.txt').exists():
        st.error("ERROR: cookies.txt file not found! Please export cookies first.")
        return None

    ydl_opts = {
        'outtmpl': f'{output_path}/input_vid.%(ext)s',
        'format': 'best[height<=720]',
        'cookiefile': './cookies.txt',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            metadata = {
                "Author": info.get('uploader', 'Unknown'),
                "Title": info.get('title', 'Unknown'),
                "Views": info.get('view_count', 0)
            }
            ydl.download([url])
        return metadata
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None

def video_to_images(video_path, output_folder):
    """Extract images from videos with better coverage for long videos."""
    clip = VideoFileClip(video_path)
    duration = clip.duration

    if duration > 300:  # 5 minutes
        fps_rate = 0.5  # 1 frame every 2 seconds for long videos
    else:
        fps_rate = 0.2  # 1 frame every 5 seconds for short videos

    clip.write_images_sequence(
        os.path.join(output_folder, "frame_%04d.png"), fps=fps_rate
    )
    return int(duration * fps_rate)

def video_to_audio(video_path, output_audio_path):
    """Extract audio from videos."""
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)

def audio_to_text(audio_path, api_key):
    """Convert audio to text using Deepgram Speech Recognition."""
    try:
        deepgram = DeepgramClient(api_key)
        
        with open(audio_path, "rb") as audio_file:
            buffer_data = audio_file.read()

        options = PrerecordedOptions(
            model="nova-2",
            language="en-US",
            punctuate=True,
            smart_format=True,
            paragraphs=True,
            utterances=True
        )

        payload = {"buffer": buffer_data}
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception as e:
        st.error(f"Could not request results from Deepgram service: {e}")
        return ""

def plot_images_streamlit(image_paths):
    """Plot images in a dynamic grid for Streamlit."""
    valid_images = [img_path for img_path in image_paths if os.path.isfile(img_path)]
    if not valid_images:
        st.warning("No valid images found to display")
        return

    valid_images = valid_images[:12]
    num_images = len(valid_images)

    if num_images <= 6:
        cols = 3
    elif num_images <= 8:
        cols = 4
    else:
        cols = 4

    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    if rows == 1:
        axes = [axes]
    
    for i, img_path in enumerate(valid_images):
        try:
            row_idx = i // cols
            col_idx = i % cols
            ax = axes[row_idx][col_idx] if rows > 1 else axes[col_idx]
            
            image = Image.open(img_path)
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(f"Frame {i+1}", fontsize=10)
        except Exception as e:
            st.error(f"Error loading image {img_path}: {e}")

    plt.tight_layout()
    st.pyplot(fig)
    st.success(f"Displayed {len(valid_images)} images from video")

def clean_up_files():
    """Remove temporary files."""
    import glob
    for pattern in ['./video_data/*', './mixed_data/*']:
        for old_file in glob.glob(pattern):
            try:
                os.remove(old_file)
            except:
                pass

def analyze_video(url, query, deepgram_key, gemini_key):
    """Main function to process and analyze the video."""
    # Clean up old files
    clean_up_files()
    
    with st.status("Processing video...", expanded=True) as status:
        # Step 1: Download video
        st.write("📥 Downloading video...")
        metadata_vid = download_video(url, output_video_path)
        if not metadata_vid:
            return None, None
        
        # Verify download
        if not os.path.exists(filepath):
            st.error("Video file not found after download")
            return None, None

        file_size = os.path.getsize(filepath)
        if file_size < 1000:
            st.error(f"Video file too small ({file_size} bytes) - download failed")
            return None, None

        st.success(f"✅ Downloaded: {metadata_vid['Title']} (Size: {file_size:,} bytes)")

        # Step 2: Process video
        st.write("🎬 Processing video frames...")
        expected_frames = video_to_images(filepath, output_folder)
        st.write(f"🖼️ Extracted ~{expected_frames} frames")

        st.write("🎵 Extracting audio...")
        video_to_audio(filepath, output_audio_path)

        st.write("📝 Transcribing audio...")
        text_data = audio_to_text(output_audio_path, deepgram_key)

        with open(output_folder + "output_text.txt", "w", encoding='utf-8') as file:
            file.write(text_data)

        if text_data.startswith("["):
            st.warning(f"⚠️ Transcription status: {text_data}")
        else:
            st.success(f"✅ Transcription successful: {len(text_data)} characters")

        # Cleanup audio file
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)

        # Step 3: Build index
        st.write("🔍 Building search index...")
        documents = SimpleDirectoryReader(output_folder).load_data()
        
        index = MultiModalVectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        retriever_engine = index.as_retriever(
            similarity_top_k=5, image_similarity_top_k=25
        )

        # Step 4: Retrieve relevant content
        st.write("🎯 Retrieving relevant content...")
        retrieval_results = retriever_engine.retrieve(query)
        
        retrieved_image = []
        retrieved_text = []
        for res_node in retrieval_results:
            if hasattr(res_node.node, 'metadata') and "file_path" in res_node.node.metadata:
                retrieved_image.append(res_node.node.metadata["file_path"])
            else:
                retrieved_text.append(res_node.text)

        # Prepare for analysis
        context_str = "".join(retrieved_text)
        
        # Get transcript
        try:
            with open(output_folder + "output_text.txt", "r", encoding='utf-8') as file:
                deepgram_transcript = file.read()

            if deepgram_transcript.strip() and not deepgram_transcript.startswith("["):
                enhanced_context = context_str + f"\n\n=== FULL VIDEO TRANSCRIPT ===\n{deepgram_transcript}\n"
                st.success(f"✅ Added Deepgram transcript ({len(deepgram_transcript)} characters)")
            else:
                enhanced_context = context_str
                st.warning("⚠️ No valid transcript - using visual analysis only")
        except:
            enhanced_context = context_str
            st.warning("⚠️ Could not read transcript - using visual analysis only")

        # Step 5: Generate analysis
        st.write("🤖 Generating AI analysis...")
        
        enhanced_qa_tmpl = (
            "You are an expert video analyst. Analyze this video comprehensively using the visual frames and transcript. "
            "Generate estimated timestamps in [MM:SS] format based on video length and content flow.\n\n"
            "INSTRUCTIONS:\n"
            "- Provide comprehensive analysis (1200+ words)\n"
            "- Generate estimated timestamps like [02:15] based on content progression\n"
            "- Quote from the transcript and reference specific visual elements\n"
            "- Provide detailed explanations and context\n\n"
            "Available Context: {context_str}\n"
            "Video Metadata: {metadata_str}\n"
            "Query: {query_str}\n\n"
            "Comprehensive Analysis with Estimated Timestamps:"
        )

        metadata_str = json.dumps(metadata_vid)
        
        # Initialize Gemini
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=6000,
                temperature=0.3,
            )
        )

        full_prompt = enhanced_qa_tmpl.format(
            context_str=enhanced_context,
            query_str=query,
            metadata_str=metadata_str
        )

        # Prepare images (limit to avoid token limits)
        images = []
        for img_path in retrieved_image[:10]:
            if os.path.isfile(img_path):
                image = Image.open(img_path)
                images.append(image)

        # Generate response
        if images:
            content = [full_prompt] + images
        else:
            content = [full_prompt]

        response = model.generate_content(content)
        status.update(label="Analysis complete!", state="complete")
        
        return response.text, retrieved_image

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="YouTube Video Analysis Tool")

# Sidebar for API keys
with st.sidebar:
    st.title("API Configuration")
    deepgram_key = st.text_input("Deepgram API Key", type="password")
    gemini_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.info("""
        **Instructions:**
        1. Enter your API keys
        2. Paste a YouTube URL
        3. Enter your query
        4. Click 'Analyze Video'
    """)

# Main content
st.title("🎥 YouTube Video Analysis Tool")
st.markdown("Analyze YouTube videos with multi-modal RAG using Gemini and Deepgram")

# Input form
with st.form("video_analysis_form"):
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    query = st.text_area(
        "Analysis Query",
        value="Using examples from video, explain all things covered in the video and give approximate timestamps of what each portion talks about",
        height=100
    )
    submitted = st.form_submit_button("🔍 Analyze Video")

# Process form submission
if submitted:
    if not url or not query:
        st.warning("Please enter both a YouTube URL and a query")
    elif not deepgram_key or not gemini_key:
        st.warning("Please enter both API keys in the sidebar")
    else:
        # Store in session state
        if 'analysis_result' not in st.session_state:
            st.session_state.analysis_result = None
        if 'retrieved_images' not in st.session_state:
            st.session_state.retrieved_images = None

        # Run analysis
        analysis_result, retrieved_images = analyze_video(url, query, deepgram_key, gemini_key)
        
        if analysis_result:
            st.session_state.analysis_result = analysis_result
            st.session_state.retrieved_images = retrieved_images

# Display results if available
if 'analysis_result' in st.session_state and st.session_state.analysis_result:
    st.markdown("---")
    st.subheader("Analysis Results")
    st.write(st.session_state.analysis_result)
    
    if 'retrieved_images' in st.session_state and st.session_state.retrieved_images:
        st.subheader("Key Video Frames Analyzed")
        plot_images_streamlit(st.session_state.retrieved_images)
