import os
import json
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import ffmpeg
from scenedetect import detect, ContentDetector
import cv2

# OpenAI Whisper for local transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ OpenAI Whisper available for local transcription")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ OpenAI Whisper not available - install openai-whisper for local transcription")

IAB_CATEGORIES = {
    # Tier 1 Categories
    "IAB1": "Arts & Entertainment",
    "IAB2": "Automotive", 
    "IAB3": "Business",
    "IAB4": "Careers",
    "IAB5": "Education",
    "IAB6": "Family & Parenting",
    "IAB7": "Health & Fitness",
    "IAB8": "Food & Drink",
    "IAB9": "Hobbies & Interests",
    "IAB10": "Home & Garden",
    "IAB11": "Law, Government & Politics",
    "IAB12": "News",
    "IAB13": "Personal Finance",
    "IAB14": "Society",
    "IAB15": "Science",
    "IAB16": "Pets",
    "IAB17": "Sports",
    "IAB18": "Style & Fashion",
    "IAB19": "Technology & Computing",
    "IAB20": "Travel",
    "IAB21": "Real Estate",
    "IAB22": "Shopping",
    "IAB23": "Religion & Spirituality",
    "IAB24": "Uncategorized",
    "IAB25": "Non-Standard Content",
    "IAB26": "Illegal Content"
}

# Common category mappings for validation
CATEGORY_MAPPINGS = {
    # Sports variations
    "sports & recreation": "IAB17",
    "sports and recreation": "IAB17", 
    "recreation": "IAB17",
    "athletics": "IAB17",
    "fitness": "IAB7",
    "exercise": "IAB7",
    
    # Entertainment variations
    "entertainment": "IAB1",
    "movies": "IAB1",
    "television": "IAB1",
    "tv": "IAB1",
    "music": "IAB1",
    "gaming": "IAB9",
    "games": "IAB9",
    
    # Technology variations
    "technology": "IAB19",
    "tech": "IAB19",
    "computing": "IAB19",
    "software": "IAB19",
    "internet": "IAB19",
    
    # Business variations
    "business": "IAB3",
    "finance": "IAB13",
    "financial": "IAB13",
    "money": "IAB13",
    "investing": "IAB13",
    
    # Education variations
    "education": "IAB5",
    "learning": "IAB5",
    "tutorial": "IAB5",
    "training": "IAB5",
    
    # Health variations
    "health": "IAB7",
    "medical": "IAB7",
    "wellness": "IAB7",
    "healthcare": "IAB7",
    
    # News variations
    "news": "IAB12",
    "current events": "IAB12",
    "politics": "IAB11",
    "government": "IAB11",
    
    # Travel variations
    "travel": "IAB20",
    "tourism": "IAB20",
    "vacation": "IAB20",
    
    # Food variations
    "food": "IAB8",
    "cooking": "IAB8",
    "recipes": "IAB8",
    "dining": "IAB8",
    "restaurants": "IAB8",
    
    # Fashion variations
    "fashion": "IAB18",
    "style": "IAB18",
    "clothing": "IAB18",
    "beauty": "IAB18",
    
    # Home variations
    "home": "IAB10",
    "garden": "IAB10",
    "diy": "IAB10",
    "interior design": "IAB10",
    
    # Automotive variations
    "automotive": "IAB2",
    "cars": "IAB2",
    "vehicles": "IAB2",
    "auto": "IAB2",
    
    # Family variations
    "family": "IAB6",
    "parenting": "IAB6",
    "kids": "IAB6",
    "children": "IAB6",
    
    # Pets variations
    "pets": "IAB16",
    "animals": "IAB16",
    "dogs": "IAB16",
    "cats": "IAB16",
    
    # Shopping variations
    "shopping": "IAB22",
    "retail": "IAB22",
    "ecommerce": "IAB22",
    
    # Real Estate variations
    "real estate": "IAB21",
    "property": "IAB21",
    "housing": "IAB21",
    
    # Religion variations
    "religion": "IAB23",
    "spiritual": "IAB23",
    "faith": "IAB23",
    
    # Science variations
    "science": "IAB15",
    "research": "IAB15",
    "scientific": "IAB15",
    
    # Society variations
    "society": "IAB14",
    "social": "IAB14",
    "culture": "IAB14",
    "community": "IAB14"
}

def validate_and_map_iab_category(category_input: str) -> Dict[str, Any]:
    """
    Validate and map input category to official IAB category.
    Returns IAB code, official name, confidence, and mapping info.
    """
    if not category_input or not isinstance(category_input, str):
        return {
            "iab_code": "IAB24",
            "iab_name": "Uncategorized",
            "confidence": 0.0,
            "original_input": category_input,
            "mapping_method": "default_fallback"
        }
    
    category_lower = category_input.lower().strip()
    
    # Check if it's already a valid IAB code
    if category_input.upper() in IAB_CATEGORIES:
        return {
            "iab_code": category_input.upper(),
            "iab_name": IAB_CATEGORIES[category_input.upper()],
            "confidence": 1.0,
            "original_input": category_input,
            "mapping_method": "direct_iab_code"
        }
    
    # Check if it's already a valid IAB category name
    for code, name in IAB_CATEGORIES.items():
        if name.lower() == category_lower:
            return {
                "iab_code": code,
                "iab_name": name,
                "confidence": 1.0,
                "original_input": category_input,
                "mapping_method": "direct_iab_name"
            }
    
    # Check direct mappings
    if category_lower in CATEGORY_MAPPINGS:
        iab_code = CATEGORY_MAPPINGS[category_lower]
        return {
            "iab_code": iab_code,
            "iab_name": IAB_CATEGORIES[iab_code],
            "confidence": 0.9,
            "original_input": category_input,
            "mapping_method": "direct_mapping"
        }
    
    # Fuzzy matching for partial matches
    best_match = None
    best_score = 0.0
    
    for mapping_key, iab_code in CATEGORY_MAPPINGS.items():
        # Check if any word in the input matches mapping key
        input_words = set(category_lower.split())
        mapping_words = set(mapping_key.split())
        
        # Calculate word overlap score
        overlap = len(input_words.intersection(mapping_words))
        total_words = len(input_words.union(mapping_words))
        
        if total_words > 0:
            score = overlap / total_words
            if score > best_score and score >= 0.5:  # At least 50% word overlap
                best_score = score
                best_match = iab_code
    
    if best_match:
        return {
            "iab_code": best_match,
            "iab_name": IAB_CATEGORIES[best_match],
            "confidence": best_score * 0.8,  # Reduce confidence for fuzzy matches
            "original_input": category_input,
            "mapping_method": "fuzzy_matching"
        }
    
    # Keyword-based fallback matching
    keyword_matches = {
        "sport": "IAB17", "game": "IAB17", "play": "IAB17",
        "tech": "IAB19", "computer": "IAB19", "digital": "IAB19",
        "business": "IAB3", "work": "IAB3", "professional": "IAB3",
        "health": "IAB7", "medical": "IAB7", "doctor": "IAB7",
        "food": "IAB8", "eat": "IAB8", "cook": "IAB8",
        "travel": "IAB20", "trip": "IAB20", "vacation": "IAB20",
        "news": "IAB12", "report": "IAB12", "journalism": "IAB12",
        "education": "IAB5", "learn": "IAB5", "teach": "IAB5",
        "entertainment": "IAB1", "fun": "IAB1", "show": "IAB1"
    }
    
    for keyword, iab_code in keyword_matches.items():
        if keyword in category_lower:
            return {
                "iab_code": iab_code,
                "iab_name": IAB_CATEGORIES[iab_code],
                "confidence": 0.6,
                "original_input": category_input,
                "mapping_method": "keyword_fallback"
            }
    
    # Default fallback
    return {
        "iab_code": "IAB24",
        "iab_name": "Uncategorized", 
        "confidence": 0.3,
        "original_input": category_input,
        "mapping_method": "uncategorized_fallback"
    }

# Configuration constants
MAX_VIDEO_DURATION = 15 * 60  # 15 minutes in seconds
MAX_FILE_SIZE_MB = 100
MAX_FRAMES_EXTRACTED = 10  # Maximum number of frames to extract per video

# Frame extraction optimization settings
FRAME_RESIZE_WIDTH = 512  # Optimal for AI analysis
FRAME_RESIZE_HEIGHT = 512
FRAME_QUALITY = 75  # JPEG quality (70-85 is optimal for AI)
ENABLE_FRAME_OPTIMIZATION = True  # Set to False to keep original quality

# Whisper transcription settings
ENABLE_LOCAL_TRANSCRIPTION = os.getenv('ENABLE_LOCAL_TRANSCRIPTION', 'true').lower() == 'true'
WHISPER_MODEL_PATH = os.getenv('WHISPER_MODEL_PATH', None)  # Path to custom model file

PORT = int(os.getenv('PORT', 3001))
WEBHOOK_URL = os.getenv('WEBHOOK_URL')

# Initialize FastAPI app
app = FastAPI(title="IAB Video Analyzer API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists
uploads_dir = Path(__file__).parent / "uploads"
frames_dir = uploads_dir / "frames"
audio_dir = uploads_dir / "audio"
models_dir = uploads_dir / "models"

uploads_dir.mkdir(exist_ok=True)
frames_dir.mkdir(exist_ok=True)
audio_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)

# Store upload progress
upload_progress: Dict[str, Dict[str, Any]] = {}

# Initialize Whisper model
whisper_model = None
if WHISPER_AVAILABLE and ENABLE_LOCAL_TRANSCRIPTION:
    try:
        # Set cache directory to a writable location
        whisper_cache_dir = models_dir / "whisper_cache"
        whisper_cache_dir.mkdir(exist_ok=True)
        
        # Set environment variable for Whisper cache directory
        os.environ['XDG_CACHE_HOME'] = str(whisper_cache_dir.parent)
        
        # Use base model by default, or custom model if specified
        model_name = WHISPER_MODEL_PATH if WHISPER_MODEL_PATH else "base"
        
        print(f"🎙️ Loading OpenAI Whisper model: {model_name}")
        print(f"📁 Using cache directory: {whisper_cache_dir}")
        whisper_model = whisper.load_model(model_name, download_root=str(whisper_cache_dir))
        print("✅ Whisper model loaded successfully for local transcription")
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        whisper_model = None
        ENABLE_LOCAL_TRANSCRIPTION = False

def cleanup_uploaded_file(video_path: Path, metadata_path: Path, upload_id: str):
    """Clean up uploaded files"""
    try:
        if video_path.exists():
            video_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        upload_progress.pop(upload_id, None)
        print(f"🗑️ Cleaned up files for upload {upload_id}")
    except Exception as error:
        print(f"❌ Failed to cleanup files for {upload_id}: {error}")

def optimize_frame_for_ai(frame, target_width=FRAME_RESIZE_WIDTH, target_height=FRAME_RESIZE_HEIGHT):
    """Optimize frame for AI analysis by resizing and maintaining aspect ratio"""
    if not ENABLE_FRAME_OPTIMIZATION:
        return frame
    
    h, w = frame.shape[:2]
    
    # Calculate scaling to fit within target dimensions while maintaining aspect ratio
    scale = min(target_width / w, target_height / h)
    
    if scale < 1.0:  # Only downscale, never upscale
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"📐 Resized frame from {w}x{h} to {new_w}x{new_h} (scale: {scale:.2f})")
    
    return frame

def extract_frames_with_scenedetect(video_path: Path, upload_id: str, original_name: str) -> Dict[str, Any]:
    """Extract frames using PySceneDetect for better scene detection"""
    print(f"🎬 Starting frame extraction with scene detection for {original_name}")
    
    frame_output_dir = frames_dir / upload_id
    frame_output_dir.mkdir(exist_ok=True)
    
    try:
        # Use PySceneDetect to find scene cuts
        scene_list = detect(str(video_path), ContentDetector())
        
        # If no scenes detected, extract frames every 5 seconds as fallback
        if not scene_list:
            print(f"⚠️ No scenes detected for {original_name}, using time-based extraction")
            return extract_frames_fallback(video_path, upload_id, original_name)
        
        # Limit scenes to MAX_FRAMES_EXTRACTED, selecting evenly distributed scenes
        if len(scene_list) > MAX_FRAMES_EXTRACTED:
            print(f"📊 Found {len(scene_list)} scenes, selecting {MAX_FRAMES_EXTRACTED} evenly distributed scenes")
            # Select evenly distributed scenes
            step = len(scene_list) / MAX_FRAMES_EXTRACTED
            selected_scenes = [scene_list[int(i * step)] for i in range(MAX_FRAMES_EXTRACTED)]
        else:
            selected_scenes = scene_list
        
        # Extract one frame from each selected scene
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_files = []
        total_size_saved = 0
        
        for i, (start_time, end_time) in enumerate(selected_scenes):
            # Get frame at the middle of the scene
            mid_time = start_time + (end_time - start_time) / 2
            frame_number = int(mid_time.get_seconds() * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Optimize frame for AI analysis
                optimized_frame = optimize_frame_for_ai(frame)
                
                frame_filename = f"scene_{i+1:03d}.jpg"
                frame_path = frame_output_dir / frame_filename
                
                # Use optimized quality setting
                cv2.imwrite(str(frame_path), optimized_frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])
                frame_files.append(frame_filename)
                
                # Track size savings
                if ENABLE_FRAME_OPTIMIZATION:
                    file_size = frame_path.stat().st_size
                    total_size_saved += file_size
        
        cap.release()
        
        optimization_info = ""
        if ENABLE_FRAME_OPTIMIZATION:
            avg_size_kb = (total_size_saved / len(frame_files) / 1024) if frame_files else 0
            optimization_info = f" (avg {avg_size_kb:.1f}KB/frame, quality={FRAME_QUALITY}%)"
        
        print(f"✅ Extracted {len(frame_files)} optimized scene-based frames for {original_name}{optimization_info}")
        return {
            "success": True,
            "frameCount": len(frame_files),
            "frameDir": str(frame_output_dir),
            "frames": [str(frame_output_dir / f) for f in frame_files],
            "optimization": {
                "enabled": ENABLE_FRAME_OPTIMIZATION,
                "target_resolution": f"{FRAME_RESIZE_WIDTH}x{FRAME_RESIZE_HEIGHT}",
                "quality": FRAME_QUALITY,
                "avg_size_kb": avg_size_kb if ENABLE_FRAME_OPTIMIZATION else None
            }
        }
    
    except Exception as error:
        print(f"❌ Scene detection failed for {original_name}: {error}")
        # Fallback to time-based extraction
        return extract_frames_fallback(video_path, upload_id, original_name)

def extract_frames_fallback(video_path: Path, upload_id: str, original_name: str) -> Dict[str, Any]:
    """Fallback frame extraction using FFmpeg (time-based, limited to MAX_FRAMES_EXTRACTED)"""
    print(f"🎬 Using fallback frame extraction for {original_name}")
    
    frame_output_dir = frames_dir / upload_id
    frame_output_dir.mkdir(exist_ok=True)
    
    try:
        # Get video duration first to calculate optimal frame extraction interval
        probe = ffmpeg.probe(str(video_path))
        duration = float(probe['format'].get('duration', 0))
        
        if duration <= 0:
            print(f"⚠️ Could not determine video duration for {original_name}, using default interval")
            extraction_rate = '1/5'  # Default: 1 frame every 5 seconds
        else:
            # Calculate interval to get approximately MAX_FRAMES_EXTRACTED frames
            optimal_interval = max(1, duration / MAX_FRAMES_EXTRACTED)
            extraction_rate = f'1/{optimal_interval}'
            print(f"📊 Video duration: {duration}s, extracting 1 frame every {optimal_interval:.1f}s to get ~{MAX_FRAMES_EXTRACTED} frames")
        
        output_pattern = str(frame_output_dir / "frame_%03d.jpg")
        
        # Build FFmpeg command with optimization settings
        input_stream = ffmpeg.input(str(video_path))
        stream = input_stream.filter('fps', fps=extraction_rate)
        
        # Add optimization filters if enabled
        if ENABLE_FRAME_OPTIMIZATION:
            # Scale video to target resolution while maintaining aspect ratio
            stream = stream.filter('scale', 
                                   width=f'min({FRAME_RESIZE_WIDTH},iw)', 
                                   height=f'min({FRAME_RESIZE_HEIGHT},ih)',
                                   force_original_aspect_ratio='decrease')
            
            # Use optimized quality settings for AI analysis
            output_args = {'q:v': 7}  # q:v 7 ≈ 75% JPEG quality
            print(f"🔧 Using optimized extraction: max {FRAME_RESIZE_WIDTH}x{FRAME_RESIZE_HEIGHT}, quality={FRAME_QUALITY}%")
        else:
            # Use original high quality settings
            output_args = {'q:v': 2}  # High quality
            print(f"🔧 Using high quality extraction (original resolution)")
        
        # Extract frames
        (
            stream
            .output(output_pattern, **output_args)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        # Count extracted frames
        frame_files = [f for f in frame_output_dir.glob("*.jpg")]
        
        # If we extracted more than MAX_FRAMES_EXTRACTED, keep only the first MAX_FRAMES_EXTRACTED
        if len(frame_files) > MAX_FRAMES_EXTRACTED:
            print(f"📊 Extracted {len(frame_files)} frames, keeping first {MAX_FRAMES_EXTRACTED}")
            # Sort frames and keep only the first MAX_FRAMES_EXTRACTED
            frame_files = sorted(frame_files)[:MAX_FRAMES_EXTRACTED]
            # Remove excess frames
            for frame_file in sorted(frame_output_dir.glob("*.jpg"))[MAX_FRAMES_EXTRACTED:]:
                frame_file.unlink()
        
        # Calculate optimization statistics
        optimization_info = ""
        avg_size_kb = None
        if frame_files and ENABLE_FRAME_OPTIMIZATION:
            total_size = sum(f.stat().st_size for f in frame_files)
            avg_size_kb = total_size / len(frame_files) / 1024
            optimization_info = f" (avg {avg_size_kb:.1f}KB/frame, optimized)"
        elif frame_files:
            total_size = sum(f.stat().st_size for f in frame_files)
            avg_size_kb = total_size / len(frame_files) / 1024
            optimization_info = f" (avg {avg_size_kb:.1f}KB/frame, original quality)"
        
        print(f"✅ Extracted {len(frame_files)} frames for {original_name}{optimization_info}")
        return {
            "success": True,
            "frameCount": len(frame_files),
            "frameDir": str(frame_output_dir),
            "frames": [str(f) for f in frame_files],
            "optimization": {
                "enabled": ENABLE_FRAME_OPTIMIZATION,
                "target_resolution": f"{FRAME_RESIZE_WIDTH}x{FRAME_RESIZE_HEIGHT}" if ENABLE_FRAME_OPTIMIZATION else "original",
                "quality": FRAME_QUALITY if ENABLE_FRAME_OPTIMIZATION else 95,
                "avg_size_kb": avg_size_kb
            }
        }
    
    except ffmpeg.Error as error:
        print(f"❌ Frame extraction failed for {original_name}: {error.stderr.decode()}")
        raise Exception(f"FFmpeg frame extraction failed: {error.stderr.decode()}")

def transcribe_audio_local(audio_path: Path, original_name: str) -> Dict[str, Any]:
    """Transcribe audio using local Whisper model"""
    if not whisper_model or not ENABLE_LOCAL_TRANSCRIPTION:
        return {
            "success": False,
            "error": "Local transcription not available",
            "transcription": None,
            "confidence": 0.0,
            "language": None,
            "segments": []
        }
    
    try:
        print(f"🎙️ Starting local transcription for {original_name}")
        
        # Transcribe audio file
        result = whisper_model.transcribe(str(audio_path))
        
        # Extract segments with timestamps
        segments = []
        if 'segments' in result and result['segments']:
            for segment in result['segments']:
                segments.append({
                    "start": segment.get('start', 0.0),
                    "end": segment.get('end', 0.0),
                    "text": segment.get('text', '').strip(),
                    "confidence": segment.get('avg_logprob', 0.0)
                })
        
        # Get full transcription text
        transcription_text = result.get('text', '').strip()
        
        # Calculate average confidence
        avg_confidence = 0.0
        if segments:
            avg_confidence = sum(s.get('confidence', 0.0) for s in segments) / len(segments)
        
        # Detect language (Whisper usually detects this)
        detected_language = result.get('language', 'en')
        
        print(f"✅ Local transcription completed for {original_name}")
        print(f"   📝 Text length: {len(transcription_text)} characters")
        print(f"   🎯 Confidence: {avg_confidence:.2f}")
        print(f"   🌍 Language: {detected_language}")
        print(f"   📊 Segments: {len(segments)}")
        
        return {
            "success": True,
            "transcription": transcription_text,
            "confidence": avg_confidence,
            "language": detected_language,
            "segments": segments,
            "word_count": len(transcription_text.split()) if transcription_text else 0,
            "duration": segments[-1]['end'] if segments else 0.0
        }
        
    except Exception as error:
        print(f"❌ Local transcription failed for {original_name}: {error}")
        return {
            "success": False,
            "error": str(error),
            "transcription": None,
            "confidence": 0.0,
            "language": None,
            "segments": []
        }

def extract_audio(video_path: Path, upload_id: str, original_name: str) -> Dict[str, Any]:
    """Extract audio as MP3 and optionally transcribe locally"""
    print(f"🔊 Starting audio extraction for {original_name}")
    
    audio_output_path = audio_dir / f"{upload_id}.mp3"
    
    try:
        # Extract audio as MP3 using ffmpeg-python
        (
            ffmpeg
            .input(str(video_path))
            .output(str(audio_output_path), acodec='mp3', audio_bitrate='192k', ar='44100')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        audio_size = audio_output_path.stat().st_size
        print(f"✅ Extracted audio for {original_name} ({audio_size} bytes)")
        
        # Perform local transcription if enabled
        transcription_result = None
        if ENABLE_LOCAL_TRANSCRIPTION and whisper_model:
            transcription_result = transcribe_audio_local(audio_output_path, original_name)
        
        return {
            "success": True,
            "audioPath": str(audio_output_path),
            "audioSize": audio_size,
            "transcription": transcription_result
        }
    
    except ffmpeg.Error as error:
        print(f"❌ Audio extraction failed for {original_name}: {error.stderr.decode()}")
        raise Exception(f"FFmpeg audio extraction failed: {error.stderr.decode()}")

def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Get video information using ffprobe"""
    try:
        probe = ffmpeg.probe(str(video_path))
        
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        
        return {
            "duration": float(probe['format'].get('duration', 0)),
            "size": int(probe['format'].get('size', 0)),
            "bitrate": int(probe['format'].get('bit_rate', 0)),
            "video": {
                "codec": video_stream['codec_name'],
                "width": video_stream['width'],
                "height": video_stream['height'],
                "fps": eval(video_stream.get('r_frame_rate', '0/1'))
            } if video_stream else None,
            "audio": {
                "codec": audio_stream['codec_name'],
                "sampleRate": audio_stream.get('sample_rate'),
                "channels": audio_stream.get('channels')
            } if audio_stream else None
        }
    
    except Exception as error:
        raise Exception(f"Failed to get video info: {error}")

async def send_to_webhook(upload_id: str, original_name: str, video_info: Dict, frames_result: Dict, audio_result: Dict, metadata_path: Path) -> Dict[str, Any]:
    """Send processed data to webhook"""
    if not WEBHOOK_URL:
        print('⚠️ No webhook URL configured, skipping webhook send')
        return {"success": False, "error": "No webhook URL configured"}
    
    try:
        print(f"🔗 Sending processed data to webhook for {original_name}")
        
        # Read all frame files as base64
        frame_dir = Path(frames_result["frameDir"])
        frame_files = sorted([f for f in frame_dir.glob("*.jpg")])
        
        frames_data = []
        for frame_file in frame_files:
            with open(frame_file, 'rb') as f:
                frame_buffer = f.read()
                frames_data.append({
                    "filename": frame_file.name,
                    "data": base64.b64encode(frame_buffer).decode('utf-8'),
                    "mimeType": "image/jpeg"
                })
        
        # Prepare transcription data instead of audio file
        transcription_data = None
        if audio_result.get("transcription") and audio_result["transcription"]["success"]:
            transcription_data = {
                "text": audio_result["transcription"]["transcription"],
                "confidence": audio_result["transcription"]["confidence"],
                "language": audio_result["transcription"]["language"],
                "wordCount": audio_result["transcription"]["word_count"],
                "duration": audio_result["transcription"]["duration"],
                "segments": audio_result["transcription"]["segments"],
            }
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        webhook_payload = {
            "uploadId": upload_id,
            "originalName": original_name,
            "processedAt": datetime.now().isoformat(),
            "videoInfo": video_info,
            "data": {
                "frames": frames_data,
                "transcription": transcription_data,
                "metadata": metadata
            }
        }
        
        # Send to webhook and wait for response
        if transcription_data:
            print(f"🔄 Sending {len(frames_data)} frames and transcription ({transcription_data['wordCount']} words) to webhook...")
        else:
            print(f"🔄 Sending {len(frames_data)} frames to webhook (no transcription available)...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(WEBHOOK_URL, json=webhook_payload)
            response_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {"rawResponse": response.text}
            
            if response.is_success:
                print(f"✅ Webhook successful for {original_name}")
                print(f"📡 Webhook Response ({response.status_code}): {json.dumps(response_data, indent=2)}")
            else:
                print(f"❌ Webhook failed for {original_name}")
                print(f"📡 HTTP {response.status_code} {response.reason_phrase}")
                print(f"📡 Error Response: {json.dumps(response_data, indent=2)}")
                raise Exception(f"Webhook failed: HTTP {response.status_code} - {response.reason_phrase}")
            
            return {
                "success": True,
                "status": response.status_code,
                "response": response_data
            }
    
    except Exception as error:
        print(f"❌ Webhook error for {original_name}: {error}")
        return {
            "success": False,
            "error": str(error),
            "webhookUrl": WEBHOOK_URL
        }

def merge_analysis_results(webhook_response: Any, upload_id: str, original_name: str) -> Dict[str, Any]:
    """
    Intelligently merge visual and audio analysis results from webhook response.
    Supports multiple response formats and provides comprehensive analysis with IAB category validation.
    """
    print(f"🔄 Merging analysis results for {original_name}")
    
    merged_result = {
        "uploadId": upload_id,
        "originalName": original_name,
        "analyzedAt": datetime.now().isoformat(),
        "visual": {
            "analysis": None,
            "confidence": 0.0,
            "tags": [],
            "objects": [],
            "scenes": [],
            "text": None,
            "iab_categories": []
        },
        "audio": {
            "analysis": None,
            "confidence": 0.0,
            "transcription": None,
            "sentiment": None,
            "language": None,
            "speakers": 0,
            "iab_categories": []
        },
        "combined": {
            "summary": None,
            "categories": [],
            "sentiment": None,
            "confidence": 0.0,
            "insights": [],
            "iab_categories": []
        },
        "rawResponse": webhook_response
    }
    
    try:
        # Handle different response formats
        visual_data, audio_data = extract_analysis_data(webhook_response)
        
        # Debug: Print webhook response structure and specific field values
        print(f"🔍 Debug - Visual data keys: {list(visual_data.keys()) if visual_data else 'None'}")
        if visual_data:
            print(f"🔍 Visual fields - topics: {visual_data.get('topics')}, tags: {visual_data.get('tags')}, keywords: {visual_data.get('keywords')}")
        
        print(f"🔍 Debug - Audio data keys: {list(audio_data.keys()) if audio_data else 'None'}")
        if audio_data:
            print(f"🔍 Audio fields - topics: {audio_data.get('topics')}, keywords: {audio_data.get('keywords')}, tags: {audio_data.get('tags')}")
        
        # Process visual analysis
        if visual_data:
            merged_result["visual"] = process_visual_analysis(visual_data)
            print(f"✅ Processed visual analysis: {len(merged_result['visual']['tags'])} tags, confidence: {merged_result['visual']['confidence']:.2f}")
        
        # Process audio analysis  
        if audio_data:
            merged_result["audio"] = process_audio_analysis(audio_data)
            print(f"✅ Processed audio analysis: {merged_result['audio']['language'] or 'unknown'} language, confidence: {merged_result['audio']['confidence']:.2f}")
        
        # Generate combined insights
        merged_result["combined"] = generate_combined_insights(
            merged_result["visual"], 
            merged_result["audio"],
            original_name
        )
        print(f"🎯 Generated combined insights: {len(merged_result['combined']['categories'])} categories, {len(merged_result['combined']['insights'])} insights")
        
        return merged_result
        
    except Exception as error:
        print(f"❌ Error merging analysis results for {original_name}: {error}")
        merged_result["error"] = str(error)
        return merged_result

def extract_analysis_data(webhook_response: Any) -> tuple:
    """Extract visual and audio data from various webhook response formats"""
    visual_data = None
    audio_data = None
    
    try:
        # Format 1: List with separate visual/audio objects
        if isinstance(webhook_response, list):
            for item in webhook_response:
                if isinstance(item, dict):
                    if "visual_result" in item:
                        visual_data = item.get("visual_result")
                    elif "image" in item or "frames" in item:
                        visual_data = item
                    elif "audio_result" in item:
                        audio_data = item.get("audio_result")
                    elif "audio" in item or "transcription" in item:
                        audio_data = item
        
        # Format 2: Single object with visual/audio keys
        elif isinstance(webhook_response, dict):
            visual_data = webhook_response.get("visual") or webhook_response.get("visual_result") or webhook_response.get("image_analysis")
            audio_data = webhook_response.get("audio") or webhook_response.get("audio_result") or webhook_response.get("audio_analysis")
            
            # If no explicit keys, assume the whole object contains both
            if not visual_data and not audio_data:
                visual_data = webhook_response
                audio_data = webhook_response
        
        # Format 3: Direct result objects
        else:
            visual_data = webhook_response
            audio_data = webhook_response
            
    except Exception as error:
        print(f"⚠️ Error extracting analysis data: {error}")
    
    return visual_data, audio_data

def process_visual_analysis(visual_data: Dict) -> Dict[str, Any]:
    """Process and normalize visual analysis data"""
    result = {
        "analysis": None,
        "confidence": 0.0,
        "tags": [],
        "objects": [],
        "scenes": [],
        "text": None,
        "emotions": [],
        "activities": [],
        "locations": []
    }
    
    if not visual_data:
        return result
    
    try:
        # Extract main analysis text
        result["analysis"] = (
            visual_data.get("summary") or
            visual_data.get("description") or 
            visual_data.get("caption") or
            visual_data.get("analysis") or
            visual_data.get("visual_result") or 
            str(visual_data)
        )
        
        # Extract confidence score
        result["confidence"] = float(visual_data.get("confidence", 0.0))
        
        # Extract tags from nested structure
        tags = []
        tags_data = visual_data.get("tags", {})
        
        if isinstance(tags_data, dict):
            # Extract from nested tag categories
            for category in ["objects", "activities", "people", "mood", "brands_text"]:
                if category in tags_data and isinstance(tags_data[category], list):
                    tags.extend(tags_data[category][:3])  # Take first 3 from each category
        elif isinstance(tags_data, list):
            tags = tags_data
        elif isinstance(tags_data, str):
            tags = [tag.strip() for tag in tags_data.split(",")]
        
        result["tags"] = list(set(tags))  # Remove duplicates
        print(f"🔍 Visual tags extracted: {result['tags']}")
        
        # Extract detected objects
        objects = visual_data.get("objects") or visual_data.get("detections") or []
        result["objects"] = objects if isinstance(objects, list) else []
        
        # Extract scene information
        scenes = visual_data.get("scenes") or visual_data.get("scene_classification") or []
        result["scenes"] = scenes if isinstance(scenes, list) else []
        
        # Extract OCR text
        result["text"] = visual_data.get("text") or visual_data.get("ocr") or visual_data.get("extracted_text")
        
        # Extract emotions
        emotions = visual_data.get("emotions") or visual_data.get("facial_emotions") or []
        result["emotions"] = emotions if isinstance(emotions, list) else []
        
        # Extract activities
        activities = visual_data.get("activities") or visual_data.get("actions") or []
        result["activities"] = activities if isinstance(activities, list) else []
        
        # Extract locations
        locations = visual_data.get("locations") or visual_data.get("places") or []
        result["locations"] = locations if isinstance(locations, list) else []
        
        iab_categories = []
        
        if visual_data.get("iab_categories"):
            if isinstance(visual_data["iab_categories"], dict):
                # Process primary category
                primary_category = visual_data["iab_categories"].get("primary")
                if primary_category:
                    validated_category = validate_and_map_iab_category(primary_category)
                    iab_categories.append(validated_category)
                    print(f"🏷️ Visual IAB Primary: {primary_category} → {validated_category['iab_code']} ({validated_category['iab_name']})")
                
                # Process secondary categories
                secondary_categories = visual_data["iab_categories"].get("secondary", [])
                if isinstance(secondary_categories, list):
                    for category in secondary_categories[:2]:  # Limit to top 2 secondary
                        validated_category = validate_and_map_iab_category(category)
                        validated_category["confidence"] = validated_category["confidence"] * 0.8  # Lower confidence for secondary
                        iab_categories.append(validated_category)
                        print(f"🏷️ Visual IAB Secondary: {category} → {validated_category['iab_code']} ({validated_category['iab_name']})")
            elif isinstance(visual_data["iab_categories"], list):
                for category in visual_data["iab_categories"][:3]:  # Limit to top 3
                    validated_category = validate_and_map_iab_category(category)
                    iab_categories.append(validated_category)
                    print(f"🏷️ Visual IAB: {category} → {validated_category['iab_code']} ({validated_category['iab_name']})")
        
        # Fallback: try to infer IAB categories from tags/topics
        if not iab_categories and result["tags"]:
            for tag in result["tags"][:2]:  # Check top 2 tags
                validated_category = validate_and_map_iab_category(tag)
                if validated_category["confidence"] >= 0.6:  # Only include confident mappings
                    iab_categories.append(validated_category)
                    print(f"🏷️ Visual IAB (from tag): {tag} → {validated_category['iab_code']} ({validated_category['iab_name']})")
        
        result["iab_categories"] = iab_categories
        
    except Exception as error:
        print(f"⚠️ Error processing visual analysis: {error}")
    
    return result

def process_audio_analysis(audio_data: Dict) -> Dict[str, Any]:
    """Process and normalize audio analysis data"""
    result = {
        "analysis": None,
        "confidence": 0.0,
        "transcription": None,
        "sentiment": None,
        "language": None,
        "speakers": 0,
        "emotions": [],
        "topics": [],
        "keywords": []
    }
    
    if not audio_data:
        return result
    
    try:
        # Extract main analysis text
        result["analysis"] = (
            audio_data.get("summary") or
            audio_data.get("analysis") or 
            audio_data.get("audio_result") or 
            str(audio_data)
        )
        
        # Extract confidence score
        result["confidence"] = float(audio_data.get("confidence", 0.0))
        
        # Extract transcription
        result["transcription"] = (
            audio_data.get("transcription") or 
            audio_data.get("transcript") or 
            audio_data.get("text")
        )
        
        # Extract sentiment
        sentiment = audio_data.get("sentiment")
        if isinstance(sentiment, dict):
            result["sentiment"] = {
                "label": sentiment.get("label"),
                "score": float(sentiment.get("score", 0.0))
            }
        elif isinstance(sentiment, str):
            result["sentiment"] = {"label": sentiment, "score": 0.0}
        
        # Extract language
        result["language"] = audio_data.get("language") or audio_data.get("detected_language")
        
        # Extract speaker count
        result["speakers"] = int(audio_data.get("speakers", 0))
        
        # Extract emotions
        emotions = audio_data.get("emotions") or audio_data.get("vocal_emotions") or []
        result["emotions"] = emotions if isinstance(emotions, list) else []
        
        # Extract topics
        topics = audio_data.get("topics") or audio_data.get("themes") or []
        result["topics"] = topics if isinstance(topics, list) else []
        
        # Extract keywords from nested structure
        keywords = []
        tags_data = audio_data.get("tags", {})
        
        if isinstance(tags_data, dict):
            # Extract from nested tag categories
            for category in ["keywords", "descriptive"]:
                if category in tags_data and isinstance(tags_data[category], list):
                    keywords.extend(tags_data[category])
        elif isinstance(tags_data, list):
            keywords = tags_data
        elif isinstance(tags_data, str):
            keywords = [kw.strip() for kw in tags_data.split(",")]
        
        result["keywords"] = list(set(keywords))  # Remove duplicates
        print(f"🔍 Audio keywords extracted: {result['keywords']}")
        
        # Process and validate IAB categories
        iab_categories = []
        
        # Check for existing IAB category data
        if audio_data.get("iab_categories"):
            if isinstance(audio_data["iab_categories"], dict):
                # Handle structured IAB data - primary category
                primary_category = audio_data["iab_categories"].get("primary")
                if primary_category:
                    validated_category = validate_and_map_iab_category(primary_category)
                    iab_categories.append(validated_category)
                    print(f"🎵 Audio IAB Primary: {primary_category} → {validated_category['iab_code']} ({validated_category['iab_name']})")
                
                # Handle structured IAB data - secondary categories
                secondary_categories = audio_data["iab_categories"].get("secondary", [])
                if isinstance(secondary_categories, list):
                    for category in secondary_categories[:2]:  # Limit to top 2 secondary
                        validated_category = validate_and_map_iab_category(category)
                        validated_category["confidence"] = validated_category["confidence"] * 0.8  # Lower confidence for secondary
                        iab_categories.append(validated_category)
                        print(f"🎵 Audio IAB Secondary: {category} → {validated_category['iab_code']} ({validated_category['iab_name']})")
            elif isinstance(audio_data["iab_categories"], list):
                # Handle list of categories
                for category in audio_data["iab_categories"][:3]:  # Limit to top 3
                    validated_category = validate_and_map_iab_category(category)
                    iab_categories.append(validated_category)
                    print(f"🎵 Audio IAB: {category} → {validated_category['iab_code']} ({validated_category['iab_name']})")
        
        # Check for category field (common in webhook responses)
        elif audio_data.get("category"):
            category = audio_data["category"]
            validated_category = validate_and_map_iab_category(category)
            iab_categories.append(validated_category)
            print(f"🎵 Audio IAB (from category): {category} → {validated_category['iab_code']} ({validated_category['iab_name']})")
        
        # Fallback: try to infer IAB categories from topics/keywords
        if not iab_categories and result["topics"]:
            for topic in result["topics"][:2]:  # Check top 2 topics
                validated_category = validate_and_map_iab_category(topic)
                if validated_category["confidence"] >= 0.6:  # Only include confident mappings
                    iab_categories.append(validated_category)
                    print(f"🎵 Audio IAB (from topic): {topic} → {validated_category['iab_code']} ({validated_category['iab_name']})")
        
        result["iab_categories"] = iab_categories
        
    except Exception as error:
        print(f"⚠️ Error processing audio analysis: {error}")
    
    return result

def generate_combined_insights(visual: Dict, audio: Dict, original_name: str) -> Dict[str, Any]:
    """Generate combined insights from visual and audio analysis"""
    combined = {
        "summary": None,
        "categories": [],
        "sentiment": None,
        "confidence": 0.0,
        "insights": [],
        "coherence_score": 0.0,
        "content_type": "unknown"
    }
    
    try:
        # Calculate overall confidence
        visual_conf = visual.get("confidence", 0.0)
        audio_conf = audio.get("confidence", 0.0)
        combined["confidence"] = (visual_conf + audio_conf) / 2 if (visual_conf > 0 or audio_conf > 0) else 0.0
        
        # Generate summary
        visual_summary = visual.get("analysis", "")
        audio_summary = audio.get("analysis", "")
        transcription = audio.get("transcription", "")
        
        summary_parts = []
        if visual_summary:
            summary_parts.append(f"Visual: {visual_summary}")
        if audio_summary:
            summary_parts.append(f"Audio: {audio_summary}")
        if transcription:
            summary_parts.append(f"Speech: {transcription[:200]}...")
        
        combined["summary"] = " | ".join(summary_parts) if summary_parts else "No analysis available"
        
        # Combine categories
        categories = set()
        
        # Add visual categories
        categories.update(visual.get("tags", []))
        categories.update(visual.get("scenes", []))
        categories.update([obj.get("class", obj) if isinstance(obj, dict) else str(obj) for obj in visual.get("objects", [])])
        
        # Add audio categories
        categories.update(audio.get("topics", []))
        categories.update(audio.get("keywords", []))
        
        combined["categories"] = list(categories)
        
        # Merge and deduplicate IAB categories
        combined_iab_categories = []
        seen_iab_codes = set()
        
        # Add visual IAB categories
        for iab_cat in visual.get("iab_categories", []):
            if iab_cat["iab_code"] not in seen_iab_codes:
                combined_iab_categories.append(iab_cat)
                seen_iab_codes.add(iab_cat["iab_code"])
        
        # Add audio IAB categories
        for iab_cat in audio.get("iab_categories", []):
            if iab_cat["iab_code"] not in seen_iab_codes:
                combined_iab_categories.append(iab_cat)
                seen_iab_codes.add(iab_cat["iab_code"])
            else:
                # If same category from both sources, update confidence to average
                for existing_cat in combined_iab_categories:
                    if existing_cat["iab_code"] == iab_cat["iab_code"]:
                        existing_cat["confidence"] = (existing_cat["confidence"] + iab_cat["confidence"]) / 2
                        existing_cat["mapping_method"] = "visual_audio_combined"
                        break
        
        # Sort by confidence (highest first)
        combined_iab_categories.sort(key=lambda x: x["confidence"], reverse=True)
        combined["iab_categories"] = combined_iab_categories
        
        if combined_iab_categories:
            category_summary = [f"{cat['iab_code']} ({cat['confidence']:.2f})" for cat in combined_iab_categories]
            print(f"🏷️ Combined IAB categories: {category_summary}")
        
        # Determine overall sentiment
        audio_sentiment = audio.get("sentiment")
        visual_emotions = visual.get("emotions", [])
        
        if audio_sentiment and isinstance(audio_sentiment, dict):
            combined["sentiment"] = audio_sentiment
        elif visual_emotions:
            # Use most common visual emotion as sentiment
            emotion_counts = {}
            for emotion in visual_emotions:
                emotion_name = emotion.get("emotion", emotion) if isinstance(emotion, dict) else str(emotion)
                emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1
            
            if emotion_counts:
                dominant_emotion = max(emotion_counts, key=emotion_counts.get)
                combined["sentiment"] = {"label": dominant_emotion, "score": 0.5}
        
        # Generate insights
        insights = []
        
        # Content type analysis
        content_type = determine_content_type(visual, audio)
        combined["content_type"] = content_type
        insights.append(f"Content appears to be: {content_type}")
        
        # Language insights
        if audio.get("language"):
            insights.append(f"Spoken language: {audio['language']}")
        
        # Speaker insights
        speaker_count = audio.get("speakers", 0)
        if speaker_count > 0:
            insights.append(f"Number of speakers detected: {speaker_count}")
        
        # Visual-audio coherence
        coherence = calculate_coherence(visual, audio)
        combined["coherence_score"] = coherence
        if coherence > 0.7:
            insights.append("High coherence between visual and audio content")
        elif coherence < 0.3:
            insights.append("Low coherence between visual and audio content")
        
        # Object-speech alignment
        if visual.get("objects") and transcription:
            visual_objects = [obj.get("class", obj) if isinstance(obj, dict) else str(obj) for obj in visual["objects"]]
            speech_words = transcription.lower().split()
            matches = [obj for obj in visual_objects if any(word in obj.lower() for word in speech_words)]
            if matches:
                insights.append(f"Objects mentioned in speech: {', '.join(matches)}")
        
        # Sentiment analysis
        if combined.get("sentiment"):
            sentiment_label = combined["sentiment"]["label"]
            insights.append(f"Overall sentiment: {sentiment_label}")
        
        combined["insights"] = insights
        
    except Exception as error:
        print(f"⚠️ Error generating combined insights: {error}")
        combined["insights"] = [f"Error generating insights: {error}"]
    
    return combined

def determine_content_type(visual: Dict, audio: Dict) -> str:
    """Determine the type of content based on analysis"""
    visual_tags = visual.get("tags", [])
    audio_topics = audio.get("topics", [])
    transcription = audio.get("transcription", "")
    
    # Educational content indicators
    education_keywords = ["lecture", "tutorial", "lesson", "teaching", "education", "explanation"]
    if any(keyword in " ".join(visual_tags + audio_topics + [transcription]).lower() for keyword in education_keywords):
        return "educational"
    
    # Entertainment content indicators
    entertainment_keywords = ["music", "song", "entertainment", "comedy", "funny", "game"]
    if any(keyword in " ".join(visual_tags + audio_topics + [transcription]).lower() for keyword in entertainment_keywords):
        return "entertainment"
    
    # News/documentary indicators
    news_keywords = ["news", "report", "documentary", "interview", "journalist"]
    if any(keyword in " ".join(visual_tags + audio_topics + [transcription]).lower() for keyword in news_keywords):
        return "news/documentary"
    
    # Business/corporate indicators
    business_keywords = ["meeting", "presentation", "business", "corporate", "conference"]
    if any(keyword in " ".join(visual_tags + audio_topics + [transcription]).lower() for keyword in business_keywords):
        return "business/corporate"
    
    # Advertisement indicators
    ad_keywords = ["product", "buy", "sale", "advertisement", "commercial", "brand"]
    if any(keyword in " ".join(visual_tags + audio_topics + [transcription]).lower() for keyword in ad_keywords):
        return "advertisement"
    
    return "general"

def calculate_coherence(visual: Dict, audio: Dict) -> float:
    """Calculate coherence score between visual and audio content"""
    try:
        score = 0.0
        factors = 0
        
        # Check sentiment alignment
        visual_emotions = [e.get("emotion", e) if isinstance(e, dict) else str(e) for e in visual.get("emotions", [])]
        audio_sentiment = audio.get("sentiment", {}).get("label", "")
        
        if visual_emotions and audio_sentiment:
            # Simple mapping of emotions to sentiment
            positive_emotions = ["happy", "joy", "excitement", "love"]
            negative_emotions = ["sad", "angry", "fear", "disgust"]
            
            visual_sentiment = None
            for emotion in visual_emotions:
                if emotion.lower() in positive_emotions:
                    visual_sentiment = "positive"
                    break
                elif emotion.lower() in negative_emotions:
                    visual_sentiment = "negative"
                    break
            
            if visual_sentiment and audio_sentiment:
                if (visual_sentiment == "positive" and audio_sentiment.lower() in ["positive", "happy", "joy"]) or \
                   (visual_sentiment == "negative" and audio_sentiment.lower() in ["negative", "sad", "angry"]):
                    score += 1.0
                factors += 1
        
        # Check topic alignment
        visual_tags = set(tag.lower() for tag in visual.get("tags", []))
        audio_keywords = set(kw.lower() for kw in audio.get("keywords", []))
        
        if visual_tags and audio_keywords:
            overlap = len(visual_tags.intersection(audio_keywords))
            total = len(visual_tags.union(audio_keywords))
            if total > 0:
                score += overlap / total
                factors += 1
        
        # Check object-transcription alignment
        visual_objects = [obj.get("class", obj).lower() if isinstance(obj, dict) else str(obj).lower() for obj in visual.get("objects", [])]
        transcription = audio.get("transcription", "").lower()
        
        if visual_objects and transcription:
            matches = sum(1 for obj in visual_objects if obj in transcription)
            if visual_objects:
                score += matches / len(visual_objects)
                factors += 1
        
        return score / factors if factors > 0 else 0.0
        
    except Exception:
        return 0.0

async def process_video_async(video_path: Path, upload_id: str, original_name: str, metadata_path: Path):
    """Process video asynchronously"""
    print(f"🔄 BACKGROUND TASK STARTED: Starting async processing for {original_name}")
    try:
        print(f"🔄 Starting async processing for {original_name}")
        
        # Update progress
        upload_progress[upload_id] = {
            "progress": 100,
            "status": "processing",
            "message": "Getting video information...",
            "stage": "info"
        }
        
        # Get video information
        video_info = get_video_info(video_path)
        print(f"📊 Video info for {original_name}: duration={video_info['duration']}, resolution={video_info['video']['width'] if video_info['video'] else 'unknown'}x{video_info['video']['height'] if video_info['video'] else 'unknown'}")
        
        # Validate video duration
        if video_info["duration"] > MAX_VIDEO_DURATION:
            duration_minutes = round(video_info["duration"] / 60, 1)
            max_minutes = MAX_VIDEO_DURATION / 60
            error_message = f"Video duration ({duration_minutes} minutes) exceeds maximum limit of {max_minutes} minutes"
            
            print(f"❌ Duration validation failed for {original_name}: {error_message}")
            
            # Update metadata with validation error
            with open(metadata_path, 'r') as f:
                current_metadata = json.load(f)
            
            error_metadata = {
                **current_metadata,
                "processing": {
                    "status": "failed",
                    "startedAt": current_metadata.get("processing", {}).get("startedAt"),
                    "failedAt": datetime.now().isoformat(),
                    "error": error_message
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(error_metadata, f, indent=2)
            
            # Update progress with error
            upload_progress[upload_id] = {
                "progress": 100,
                "status": "error",
                "error": error_message
            }
            
            # Clean up uploaded file
            cleanup_uploaded_file(video_path, metadata_path, upload_id)
            return
        
        # Update progress
        upload_progress[upload_id] = {
            "progress": 100,
            "status": "processing",
            "message": "Extracting frames with scene detection...",
            "stage": "frames"
        }
        
        # Extract frames using PySceneDetect
        frames_result = extract_frames_with_scenedetect(video_path, upload_id, original_name)
        
        # Update progress
        upload_progress[upload_id] = {
            "progress": 100,
            "status": "processing",
            "message": "Extracting audio as MP3...",
            "stage": "audio"
        }
        
        # Extract audio as MP3
        audio_result = extract_audio(video_path, upload_id, original_name)
        
        # Update metadata with processing results
        with open(metadata_path, 'r') as f:
            current_metadata = json.load(f)
        
        # Prepare audio processing results
        audio_processing_result = {
            "path": audio_result["audioPath"],
            "size": audio_result["audioSize"],
            "format": "MP3",
            "bitrate": "192kbps",
            "sampleRate": "44.1kHz"
        }
        
        # Add transcription results if available
        if audio_result.get("transcription") and audio_result["transcription"]["success"]:
            audio_processing_result["transcription"] = {
                "text": audio_result["transcription"]["transcription"],
                "confidence": audio_result["transcription"]["confidence"],
                "language": audio_result["transcription"]["language"],
                "wordCount": audio_result["transcription"]["word_count"],
                "duration": audio_result["transcription"]["duration"],
                "segments": len(audio_result["transcription"]["segments"]),
                "method": "openai-whisper-local"
            }
            print(f"💬 Local transcription included in metadata: {audio_result['transcription']['word_count']} words")
        
        updated_metadata = {
            **current_metadata,
            "videoInfo": video_info,
            "processing": {
                "status": "completed",
                "startedAt": current_metadata.get("processing", {}).get("startedAt"),
                "completedAt": datetime.now().isoformat(),
                "results": {
                    "frames": {
                        "count": frames_result["frameCount"],
                        "directory": frames_result["frameDir"],
                        "extractionMethod": "PySceneDetect + fallback"
                    },
                    "audio": audio_processing_result
                }
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(updated_metadata, f, indent=2)
        
        # Update progress to AI analysis stage
        upload_progress[upload_id] = {
            "progress": 100,
            "status": "ai-analysis",
            "message": "Video processing completed. Starting AI analysis...",
            "stage": "ai-analysis",
            "results": updated_metadata["processing"]["results"]
        }
        
        print(f"🎉 Processing completed for {original_name}:")
        print(f"   📸 Frames: {frames_result['frameCount']}")
        print(f"   🔊 Audio: {round(audio_result['audioSize'] / 1024)}KB MP3")
        
        # Send processed data to webhook
        print(f"🔄 Sending processed data to webhook...")
        webhook_result = await send_to_webhook(upload_id, original_name, video_info, frames_result, audio_result, metadata_path)
        
        if webhook_result["success"]:
            print(f"🎉 Webhook completed successfully for {original_name}")
            
            # Merge and process analysis results
            merged_analysis = merge_analysis_results(
                webhook_result["response"], 
                upload_id, 
                original_name
            )
            
            # Update metadata with comprehensive AI results
            try:
                with open(metadata_path, 'r') as f:
                    current_metadata = json.load(f)
                
                updated_metadata = {
                    **current_metadata,
                    "webhook": {
                        "attempted": True,
                        "success": True,
                        "completedAt": datetime.now().isoformat(),
                        "response": webhook_result["response"]
                    },
                    "aiAnalysis": merged_analysis  # Use comprehensive merged analysis
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(updated_metadata, f, indent=2)
                
                print(f"💾 Saved comprehensive AI analysis results to metadata for {original_name}")
                print(f"🎯 Analysis summary: {merged_analysis['combined']['content_type']} content, {len(merged_analysis['combined']['categories'])} categories, coherence: {merged_analysis['combined']['coherence_score']:.2f}")
                
                # Update final progress with comprehensive analysis
                upload_progress[upload_id] = {
                    "progress": 100,
                    "status": "completed",
                    "message": f"Processing and AI analysis completed! Extracted {frames_result['frameCount']} frames, audio, and comprehensive AI insights.",
                    "results": updated_metadata["processing"]["results"],
                    "aiAnalysis": merged_analysis
                }
                
            except Exception as meta_error:
                print(f"Failed to update metadata with merged AI results: {meta_error}")
                upload_progress[upload_id] = {
                    "progress": 100,
                    "status": "ai-analysis-error",
                    "message": f"Processing completed! Extracted {frames_result['frameCount']} frames and audio. AI analysis failed.",
                    "results": updated_metadata["processing"]["results"],
                    "aiAnalysisError": str(meta_error)
                }
        else:
            print(f"⚠️ Webhook failed for {original_name}: {webhook_result['error']}")
            
            # Update metadata to indicate webhook failure
            try:
                with open(metadata_path, 'r') as f:
                    current_metadata = json.load(f)
                
                updated_metadata = {
                    **current_metadata,
                    "webhook": {
                        "attempted": True,
                        "success": False,
                        "error": webhook_result["error"],
                        "attemptedAt": datetime.now().isoformat()
                    }
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(updated_metadata, f, indent=2)
                
                upload_progress[upload_id] = {
                    "progress": 100,
                    "status": "ai-analysis-error",
                    "message": f"Processing completed! Extracted {frames_result['frameCount']} frames and audio. AI analysis failed.",
                    "results": updated_metadata["processing"]["results"],
                    "aiAnalysisError": webhook_result["error"]
                }
                
            except Exception as meta_error:
                print(f"Failed to update metadata with webhook error: {meta_error}")
    
    except Exception as error:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Processing failed for {original_name}: {error}")
        print(f"📋 Full processing error traceback:\n{error_trace}")
        
        # Create more specific error messages
        error_message = str(error)
        if "No such file or directory" in error_message:
            error_message = f"Required file missing during processing: {error_message}"
        elif "Permission denied" in error_message:
            error_message = f"Permission error during processing: {error_message}"
        elif "ffmpeg" in error_message.lower():
            error_message = f"Video processing error (FFmpeg): {error_message}"
        elif "whisper" in error_message.lower():
            error_message = f"Audio transcription error (Whisper): {error_message}"
        elif "opencv" in error_message.lower() or "cv2" in error_message.lower():
            error_message = f"Video analysis error (OpenCV): {error_message}"
        elif "Memory" in error_message or "memory" in error_message:
            error_message = f"Out of memory during processing: {error_message}"
        else:
            error_message = f"Processing failed: {error_message}"
        
        # Update metadata with error
        try:
            with open(metadata_path, 'r') as f:
                current_metadata = json.load(f)
            
            error_metadata = {
                **current_metadata,
                "processing": {
                    "status": "failed",
                    "startedAt": current_metadata.get("processing", {}).get("startedAt"),
                    "failedAt": datetime.now().isoformat(),
                    "error": error_message,
                    "originalError": str(error)
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(error_metadata, f, indent=2)
        
        except Exception as meta_error:
            print(f"Failed to update metadata with error: {meta_error}")
        
        # Update progress with error
        upload_progress[upload_id] = {
            "progress": 100,
            "status": "error",
            "error": error_message
        }

# API Endpoints

@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    """Upload video endpoint with progress tracking"""
    upload_id = str(uuid.uuid4())
    
    # Initialize progress tracking
    upload_progress[upload_id] = {"progress": 0, "status": "uploading"}
    
    # Validate file type
    if not video.content_type or not video.content_type.startswith('video/'):
        upload_progress[upload_id] = {"progress": 0, "status": "error", "error": "Only video files are allowed!"}
        raise HTTPException(status_code=400, detail="Only video files are allowed!")
    
    try:
        # Save uploaded file using streaming to avoid memory issues
        file_extension = Path(video.filename).suffix if video.filename else ".mp4"
        filename = f"{upload_id}{file_extension}"
        video_path = uploads_dir / filename
        
        # Stream the file to disk instead of reading all into memory
        total_size = 0
        with open(video_path, 'wb') as f:
            while chunk := await video.read(8192):  # Read in 8KB chunks
                f.write(chunk)
                total_size += len(chunk)
                # Check file size limit as we go
                if total_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    f.close()
                    video_path.unlink()  # Delete the partial file
                    upload_progress[upload_id] = {"progress": 0, "status": "error", "error": f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB}MB"}
                    raise HTTPException(status_code=400, detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB}MB")
        
        # Create minimal file info (defer heavy processing to background)
        file_info = {
            "id": upload_id,
            "filename": filename,
            "originalname": video.filename,
            "mimetype": video.content_type,
            "size": total_size,
            "path": str(video_path),
            "uploadedAt": datetime.now().isoformat(),
            "processing": {
                "status": "started",
                "startedAt": datetime.now().isoformat()
            }
        }
        
        # Save file metadata
        metadata_path = uploads_dir / f"{upload_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(file_info, f, indent=2)
        
        # Update progress to complete upload, start processing
        upload_progress[upload_id] = {
            "progress": 100,
            "status": "processing",
            "message": "Video uploaded, starting processing...",
            "fileInfo": file_info
        }
        
        print(f"✅ Video uploaded successfully: {video.filename} ({total_size} bytes)")
        
        # Start processing asynchronously
        print(f"🚀 About to start background task for {video.filename}")
        background_tasks.add_task(process_video_async, video_path, upload_id, video.filename, metadata_path)
        print(f"🚀 Background task added, about to return response for {video.filename}")
        
        return {
            "message": "Video uploaded successfully, processing started",
            "uploadId": upload_id,
            "fileInfo": {
                **file_info,
                "processing": {"status": "started"}
            }
        }
    
    except HTTPException as http_error:
        # Re-raise HTTPExceptions (like file size errors) as-is
        print(f"❌ HTTP error during upload: {http_error.detail}")
        upload_progress[upload_id] = {
            "progress": 0,
            "status": "error",
            "error": http_error.detail
        }
        raise http_error
    except Exception as error:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error saving file: {error}")
        print(f"📋 Full traceback:\n{error_trace}")
        
        # Create more specific error messages
        error_message = str(error)
        if "No space left on device" in error_message:
            error_message = "Server storage is full. Please try again later."
        elif "Permission denied" in error_message:
            error_message = "Server permission error. Please contact support."
        elif "File name too long" in error_message:
            error_message = "File name is too long. Please rename your file."
        elif "Invalid" in error_message.lower():
            error_message = f"Invalid file: {error_message}"
        else:
            error_message = f"Upload failed: {error_message}"
        
        upload_progress[upload_id] = {
            "progress": 0,
            "status": "error",
            "error": error_message
        }
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/api/upload/progress/{upload_id}")
async def get_upload_progress(upload_id: str):
    """Get upload progress"""
    progress = upload_progress.get(upload_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Upload not found")
    return progress

@app.get("/api/files/{upload_id}")
async def get_file_info(upload_id: str):
    """Get file info endpoint"""
    try:
        metadata_path = uploads_dir / f"{upload_id}_metadata.json"
        with open(metadata_path, 'r') as f:
            file_info = json.load(f)
        return file_info
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/files/{upload_id}/status")
async def get_processing_status(upload_id: str):
    """Get processing status endpoint"""
    try:
        # Check progress first
        progress = upload_progress.get(upload_id)
        if progress:
            return progress
        
        # If not in progress, check metadata file
        metadata_path = uploads_dir / f"{upload_id}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if metadata.get("processing"):
            processing = metadata["processing"]
            frame_count = processing.get('results', {}).get('frames', {}).get('count', 0)
            
            if processing.get("status") == "completed":
                message = f"Processing completed! Extracted {frame_count} frames and audio."
            else:
                message = processing.get("error", "Processing status unknown")
            
            return {
                "progress": 100,
                "status": processing.get("status", "unknown"),
                "message": message,
                "results": processing.get("results"),
                "aiAnalysis": metadata.get("aiAnalysis"),
                "aiAnalysisError": metadata.get("webhook", {}).get("error")
            }
        else:
            return {
                "progress": 100,
                "status": "completed",
                "message": "File uploaded (no processing information available)"
            }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Upload not found")

@app.get("/api/files/{upload_id}/video")
async def serve_video(upload_id: str, request: Request):
    """Serve video files with streaming support"""
    try:
        metadata_path = uploads_dir / f"{upload_id}_metadata.json"
        with open(metadata_path, 'r') as f:
            file_info = json.load(f)
        
        video_path = Path(file_info["path"])
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Support for range requests (video streaming)
        range_header = request.headers.get('Range')
        if range_header:
            start, end = 0, video_path.stat().st_size - 1
            
            # Parse range header
            range_match = range_header.replace('bytes=', '').split('-')
            if range_match[0]:
                start = int(range_match[0])
            if range_match[1]:
                end = int(range_match[1])
            
            chunk_size = end - start + 1
            
            def iterfile():
                with open(video_path, 'rb') as f:
                    f.seek(start)
                    remaining = chunk_size
                    while remaining:
                        chunk = f.read(min(8192, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            
            headers = {
                'Content-Range': f'bytes {start}-{end}/{video_path.stat().st_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(chunk_size),
                'Content-Type': file_info.get('mimetype', 'video/mp4'),
            }
            
            return StreamingResponse(iterfile(), status_code=206, headers=headers)
        
        # Return full file
        return FileResponse(video_path, media_type=file_info.get('mimetype', 'video/mp4'))
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/files")
async def list_files():
    """List all uploaded files"""
    try:
        metadata_files = list(uploads_dir.glob("*_metadata.json"))
        file_list = []
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    file_info = json.load(f)
                file_list.append(file_info)
            except Exception:
                continue
        
        return file_list
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to list files")

@app.delete("/api/files/{upload_id}")
async def delete_file(upload_id: str):
    """Delete file endpoint"""
    try:
        metadata_path = uploads_dir / f"{upload_id}_metadata.json"
        with open(metadata_path, 'r') as f:
            file_info = json.load(f)
        
        # Delete video file
        video_path = Path(file_info["path"])
        if video_path.exists():
            video_path.unlink()
        
        # Delete metadata file
        metadata_path.unlink()
        
        # Delete frames directory
        frame_dir = frames_dir / upload_id
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        
        # Delete audio file
        audio_file = audio_dir / f"{upload_id}.mp3"
        if audio_file.exists():
            audio_file.unlink()
        
        # Remove from progress tracking
        upload_progress.pop(upload_id, None)
        
        return {"message": "File deleted successfully"}
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/health")
async def health_check():
    """Health check endpoint with dependency status"""
    print("💓 Health check requested")
    
    # Check dependencies
    dependencies = {
        "opencv": False,
        "ffmpeg": False,
        "whisper": WHISPER_AVAILABLE,
        "scenedetect": False,
        "httpx": False
    }
    
    # Test OpenCV
    try:
        import cv2
        dependencies["opencv"] = True
    except ImportError:
        pass
    
    # Test FFmpeg
    try:
        import ffmpeg
        dependencies["ffmpeg"] = True
    except ImportError:
        pass
    
    # Test SceneDetect
    try:
        from scenedetect import detect
        dependencies["scenedetect"] = True
    except ImportError:
        pass
    
    # Test httpx
    try:
        import httpx
        dependencies["httpx"] = True
    except ImportError:
        pass
    
    # Check directory permissions
    directory_status = {}
    for name, path in [("uploads", uploads_dir), ("frames", frames_dir), ("audio", audio_dir)]:
        try:
            # Try to create a test file
            test_file = path / "health_check_test.txt"
            test_file.write_text("test")
            test_file.unlink()
            directory_status[name] = {"writable": True, "exists": True, "path": str(path)}
        except Exception as e:
            directory_status[name] = {"writable": False, "exists": path.exists(), "path": str(path), "error": str(e)}
    
    all_deps_ok = all(dependencies.values())
    all_dirs_ok = all(d.get("writable", False) for d in directory_status.values())
    
    return {
        "status": "healthy" if (all_deps_ok and all_dirs_ok) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "uploadsDir": str(uploads_dir),
        "activeUploads": len(upload_progress),
        "dependencies": dependencies,
        "directories": directory_status,
        "features": {
            "whisperAvailable": WHISPER_AVAILABLE,
            "localTranscription": ENABLE_LOCAL_TRANSCRIPTION and whisper_model is not None,
            "frameOptimization": ENABLE_FRAME_OPTIMIZATION,
            "webhookConfigured": WEBHOOK_URL is not None
        },
        "issues": {
            "missing_dependencies": [k for k, v in dependencies.items() if not v],
            "directory_issues": [k for k, v in directory_status.items() if not v.get("writable", False)]
        },
        "config": {
            "maxFileSize": f"{MAX_FILE_SIZE_MB}MB",
            "maxDuration": f"{MAX_VIDEO_DURATION}s",
            "maxFrames": MAX_FRAMES_EXTRACTED,
            "port": PORT
        }
    }

@app.get("/api/files/{upload_id}/frames")
async def get_frames_list(upload_id: str):
    """Get list of extracted frames with optimization info"""
    try:
        frame_dir = frames_dir / upload_id
        
        if not frame_dir.exists():
            raise HTTPException(status_code=404, detail="Frames not found")
        
        frame_files = sorted([f for f in frame_dir.glob("*.jpg")])
        
        frames = []
        total_size = 0
        
        for frame_file in frame_files:
            file_size = frame_file.stat().st_size
            total_size += file_size
            
            frames.append({
                "filename": frame_file.name,
                "url": f"/api/files/{upload_id}/frames/{frame_file.name}",
                "path": str(frame_file),
                "size": file_size
            })
        
        # Calculate optimization statistics
        avg_size_kb = (total_size / len(frames) / 1024) if frames else 0
        
        # Get optimization info from metadata if available
        metadata_path = uploads_dir / f"{upload_id}_metadata.json"
        optimization_info = {
            "enabled": ENABLE_FRAME_OPTIMIZATION,
            "target_resolution": f"{FRAME_RESIZE_WIDTH}x{FRAME_RESIZE_HEIGHT}" if ENABLE_FRAME_OPTIMIZATION else "original",
            "quality": FRAME_QUALITY if ENABLE_FRAME_OPTIMIZATION else 95,
            "avg_size_kb": avg_size_kb
        }
        
        # Try to get optimization info from stored metadata
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get('processing', {}).get('results', {}).get('frames', {}).get('optimization'):
                        stored_optimization = metadata['processing']['results']['frames']['optimization']
                        optimization_info.update(stored_optimization)
            except Exception as e:
                print(f"⚠️ Could not read optimization info from metadata: {e}")
        
        return {
            "uploadId": upload_id,
            "frameCount": len(frames),
            "frames": frames,
            "optimization": optimization_info,
            "totalSize": total_size,
            "avgSizeKB": avg_size_kb
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail="Failed to list frames")

@app.get("/api/files/{upload_id}/frames/{filename}")
async def serve_frame(upload_id: str, filename: str):
    """Serve individual frame image"""
    frame_path = frames_dir / upload_id / filename
    
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    
    return FileResponse(frame_path, media_type="image/jpeg")

@app.get("/api/files/{upload_id}/audio")
async def serve_audio(upload_id: str):
    """Serve extracted audio file"""
    audio_path = audio_dir / f"{upload_id}.mp3"
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    
    return FileResponse(audio_path, media_type="audio/mpeg")

@app.get("/api/files/{upload_id}/transcription")
async def get_transcription(upload_id: str):
    """Get transcription for uploaded video"""
    try:
        metadata_path = uploads_dir / f"{upload_id}_metadata.json"
        
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Upload not found")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if transcription exists in processing results
        transcription_data = metadata.get('processing', {}).get('results', {}).get('audio', {}).get('transcription')
        
        if not transcription_data:
            # Try to get from AI analysis if available
            ai_analysis = metadata.get('aiAnalysis', {})
            audio_analysis = ai_analysis.get('audio', {})
            if audio_analysis.get('transcription'):
                transcription_data = {
                    "text": audio_analysis['transcription'],
                    "confidence": audio_analysis.get('confidence', 0.0),
                    "language": audio_analysis.get('language', 'unknown'),
                    "method": "webhook-ai-analysis"
                }
        
        if not transcription_data:
            return {
                "uploadId": upload_id,
                "transcriptionAvailable": False,
                "message": "No transcription available. Local transcription may be disabled or failed.",
                "localTranscriptionEnabled": ENABLE_LOCAL_TRANSCRIPTION and whisper_model is not None
            }
        
        return {
            "uploadId": upload_id,
            "transcriptionAvailable": True,
            "transcription": transcription_data,
            "originalName": metadata.get('originalname', 'unknown'),
            "processingMethod": transcription_data.get('method', 'unknown')
        }
        
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to get transcription: {str(error)}")

@app.post("/api/test/iab-category")
async def test_iab_category_mapping(request: Request):
    """Test IAB category mapping functionality"""
    try:
        body = await request.json()
        category_input = body.get("category", "")
        
        if not category_input:
            raise HTTPException(status_code=400, detail="Category input is required")
        
        # Test the mapping function
        result = validate_and_map_iab_category(category_input)
        
        return {
            "input": category_input,
            "mapping_result": result,
            "available_iab_categories": len(IAB_CATEGORIES),
            "available_mappings": len(CATEGORY_MAPPINGS)
        }
        
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to test IAB mapping: {str(error)}")

@app.get("/api/iab-categories")
async def get_iab_categories():
    """Get all available IAB categories"""
    return {
        "categories": IAB_CATEGORIES,
        "mappings_count": len(CATEGORY_MAPPINGS),
        "total_categories": len(IAB_CATEGORIES)
    }

@app.post("/api/test/upload")
async def test_upload_endpoint(video: UploadFile = File(...)):
    """Test upload endpoint for debugging - accepts file but doesn't process it"""
    try:
        # Validate file type
        if not video.content_type or not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Only video files are allowed!")
        
        # Read file to validate it can be uploaded
        content = await video.read(1024)  # Read first 1KB to test
        
        return {
            "status": "success",
            "message": "Upload test successful - file can be received",
            "fileInfo": {
                "filename": video.filename,
                "contentType": video.content_type,
                "size": video.size if hasattr(video, 'size') else "unknown",
                "firstBytes": len(content)
            },
            "serverReady": True
        }
        
    except Exception as error:
        import traceback
        return {
            "status": "error",
            "message": f"Upload test failed: {str(error)}",
            "error": str(error),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True) 
