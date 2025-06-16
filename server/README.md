# IAB Video Analyzer - Python Server

This is the Python/FastAPI version of the video analyzer server, rewritten from Node.js to leverage PySceneDetect for improved frame extraction.

## Features

- **FastAPI** web framework (high performance, async)
- **PySceneDetect** for intelligent scene-based frame extraction
- **FFmpeg** integration for audio extraction and video info
- **Automatic API documentation** at `/docs` (Swagger UI)
- **Background processing** with progress tracking
- **Video streaming** support with range requests
- **Webhook integration** for AI analysis

## Improvements over Node.js version

1. **Better Frame Extraction**: Uses PySceneDetect to intelligently detect scene changes and extract representative frames, instead of just extracting frames every 5 seconds.
2. **Fallback Support**: If scene detection fails, falls back to time-based extraction.
3. **Type Safety**: Uses Python type hints for better code reliability.
4. **Async Performance**: FastAPI provides excellent async performance for handling multiple uploads.
5. **Auto Documentation**: Automatic API documentation generation.

## Requirements

- Python 3.11+
- FFmpeg (for video/audio processing)
- All Python dependencies listed in `requirements.txt`

## Installation

### Local Development

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install FFmpeg:
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt install ffmpeg`
   - **Windows**: Download from https://ffmpeg.org/

3. Set environment variables:
   ```bash
   export PORT=3001
   export WEBHOOK_URL=your-webhook-url  # optional
   ```

4. Run the server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 3001 --reload
   ```

### Docker

1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

2. Or build manually:
   ```bash
   docker build -t iab-video-analyzer-py .
   docker run -p 3001:3001 -v $(pwd)/uploads:/app/uploads iab-video-analyzer-py
   ```

## API Endpoints

- `POST /api/upload` - Upload video files
- `GET /api/upload/progress/{upload_id}` - Get upload/processing progress
- `GET /api/files` - List all uploaded files
- `GET /api/files/{upload_id}` - Get file metadata
- `GET /api/files/{upload_id}/video` - Stream video file
- `DELETE /api/files/{upload_id}` - Delete uploaded file
- `GET /api/health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)

## Configuration

Environment variables:

- `PORT` - Server port (default: 3001)
- `WEBHOOK_URL` - Optional webhook URL for AI analysis
- `ENVIRONMENT` - Environment (development/production)

## Frame Extraction Methods

1. **Scene Detection (Primary)**: Uses PySceneDetect to find scene cuts and extracts one frame from the middle of each scene.
2. **Time-based (Fallback)**: If scene detection fails, extracts frames every 5 seconds using FFmpeg.

## File Structure

```
uploads/
├── frames/
│   └── {upload_id}/          # Extracted frames for each video
├── audio/
│   └── {upload_id}.mp3       # Extracted audio files
├── {upload_id}.{ext}         # Original video files
└── {upload_id}_metadata.json # File metadata and processing results
```

## Development

For development with auto-reload:

```bash
uvicorn main:app --host 0.0.0.0 --port 3001 --reload
```

The server will automatically restart when you make changes to the code. 
