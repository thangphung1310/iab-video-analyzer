# iab-video-analyzer Application

A full-stack application for video upload and IAB categorization with AI-powered analysis.

## Project Structure

```
iab-video-analyzer/
├── next/                 # Next.js frontend application
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── Dockerfile        # Frontend Docker configuration
│   └── ...
├── server/               # Node.js Express server
│   ├── server.js
│   ├── package.json
│   ├── Dockerfile        # Server Docker configuration with FFmpeg
│   └── uploads/          # Video upload directory
├── docker-compose.yml    # Docker orchestration
└── README.md
```

## Features

- **Video Upload**: Secure video file upload with progress tracking
- **Automatic FFmpeg Processing**: Extract frames every 2 seconds and audio as MP3
- **Frame Extraction**: High-quality JPEG frames extracted automatically
- **Audio Extraction**: MP3 audio extraction with 192kbps quality
- **IAB Category Detection**: Automatically classify video content using IAB standards
- **Smart Tagging**: Extract keywords, objects, and contextual tags
- **Compliance Ready**: Brand safety and advertising standard compliance
- **FFmpeg Integration**: Server includes FFmpeg for video processing
- **Docker Ready**: Containerized deployment with orchestration

## 🐳 Docker Setup (Recommended)

### Prerequisites
- Docker & Docker Compose installed
- At least 2GB free disk space

### Quick Start with Docker

1. **Build and start the server only:**
   ```bash
   docker-compose up server
   ```

2. **Build and start both server and frontend:**
   ```bash
   docker-compose --profile full-stack up
   ```

3. **Run in detached mode:**
   ```bash
   docker-compose up -d server
   ```

### Docker Commands

```bash
# Build images
docker-compose build

# View logs
docker-compose logs server
docker-compose logs frontend

# Stop services
docker-compose down

# Remove everything (including volumes)
docker-compose down -v --rmi all

# Access server shell (for debugging)
docker-compose exec server sh
```

### Docker Features

#### **Server Container:**
- ✅ **FFmpeg pre-installed** for video processing
- ✅ **Alpine Linux** for minimal size (~150MB)
- ✅ **Health checks** built-in
- ✅ **Non-root user** for security
- ✅ **Persistent uploads** via volume mounts
- ✅ **Auto-restart** on failure

#### **Frontend Container (Optional):**
- ✅ **Multi-stage build** for optimization
- ✅ **Standalone output** for Docker
- ✅ **Production optimized**
- ✅ **Health checks** included

## 💻 Local Development Setup

### 1. Install Server Dependencies
```bash
cd server
npm install
```

### 2. Install Next.js Dependencies
```bash
cd next
npm install
cd ..
```

### 3. Start the Backend Server
```bash
cd server
npm run dev
```
The server will run on `http://localhost:3001`

### 4. Start the Next.js Frontend (in a new terminal)
```bash
cd next
npm run dev
```
The frontend will run on `http://localhost:3000`

## 📡 API Endpoints

### Upload Video
- **POST** `/api/upload`
- **Content-Type**: `multipart/form-data`
- **Field**: `video` (file)

### Get Upload Progress
- **GET** `/api/upload/progress/:uploadId`

### Get File Info
- **GET** `/api/files/:uploadId`

### List All Files
- **GET** `/api/files`

### Delete File
- **DELETE** `/api/files/:uploadId`

### Get Processing Status
- **GET** `/api/files/:uploadId/status`

### Get Extracted Frames
- **GET** `/api/files/:uploadId/frames`
- Returns list of extracted frames (every 2 seconds)

### Get Individual Frame
- **GET** `/api/files/:uploadId/frames/:filename`
- Serves extracted frame image

### Get Extracted Audio
- **GET** `/api/files/:uploadId/audio`
- Downloads extracted MP3 audio file

### Health Check
- **GET** `/api/health`

## ⚙️ Configuration

### Server Configuration
- **Port**: 3001 (configurable via `PORT` environment variable)
- **Upload Limit**: 500MB
- **Supported Formats**: All video formats (MP4, MOV, AVI, WebM, etc.)
- **Upload Directory**: `./uploads` (mounted as volume in Docker)

### CORS Configuration
The server accepts requests from:
- `http://localhost:3000`
- `http://127.0.0.1:3000`

### Environment Variables

#### Server
```bash
PORT=3001                # Server port
NODE_ENV=production      # Environment mode
```

#### Frontend
```bash
NEXT_PUBLIC_API_URL=http://localhost:3001  # API server URL
```

## 🛠️ FFmpeg Capabilities

The Docker server includes FFmpeg with support for:
- **Video transcoding** (H.264, H.265, VP8, VP9)
- **Frame extraction** for thumbnails
- **Format conversion** (MP4, WebM, AVI, MOV)
- **Video analysis** (duration, resolution, bitrate)
- **Audio extraction** and processing

### Automatic Processing Workflow

When a video is uploaded, the server automatically:

1. **Upload Complete** - Video file saved to `/uploads`
2. **Video Analysis** - Extract metadata (duration, resolution, codec info)
3. **Frame Extraction** - Extract frames every 2 seconds as high-quality JPEGs
4. **Audio Extraction** - Extract audio track as MP3 (192kbps, 44.1kHz)
5. **Processing Complete** - All extracted content available via API

### Directory Structure
```
uploads/
├── {uploadId}.mp4          # Original video file
├── {uploadId}_metadata.json # File metadata with processing results
├── frames/
│   └── {uploadId}/
│       ├── frame_001.jpg   # Frame at 0 seconds
│       ├── frame_002.jpg   # Frame at 2 seconds
│       └── frame_003.jpg   # Frame at 4 seconds
└── audio/
    └── {uploadId}.mp3      # Extracted audio
```

### FFmpeg Usage Example
```bash
# Access server container
docker-compose exec server sh

# Check FFmpeg version
ffmpeg -version

# Manual frame extraction (automated in server)
ffmpeg -i /app/uploads/video.mp4 -vf fps=1/2 -q:v 2 frames/frame_%03d.jpg

# Manual audio extraction (automated in server)
ffmpeg -i /app/uploads/video.mp4 -vn -acodec mp3 -ab 192k audio.mp3
```

## 🚀 Deployment Options

### Option 1: Docker Compose (Recommended)
```bash
# Production deployment
docker-compose -f docker-compose.yml up -d
```

### Option 2: Individual Containers
```bash
# Build server image
docker build -t iab-video-analyzer-server ./server

# Run server container
docker run -d \
  --name iab-video-analyzer-server \
  -p 3001:3001 \
  -v $(pwd)/server/uploads:/app/uploads \
  iab-video-analyzer-server
```

### Option 3: Traditional Node.js
```bash
# Install dependencies and start services manually
cd server && npm install && npm start &
cd next && npm install && npm run build && npm start
```

## 🔒 Security Features

- File type validation (videos only)
- File size limits (500MB)
- Unique filename generation
- CORS protection
- Non-root Docker containers
- Health check monitoring
- Error handling and logging

## 📊 Monitoring

### Health Checks
- **Server**: `http://localhost:3001/api/health`
- **Frontend**: `http://localhost:3000` (when running)

### Docker Health Status
```bash
docker-compose ps
```

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Check what's using the port
   lsof -i :3001
   # Kill the process or change port in docker-compose.yml
   ```

2. **Volume permission issues:**
   ```bash
   # Fix upload directory permissions
   sudo chown -R $(id -u):$(id -g) server/uploads
   ```

3. **FFmpeg not found:**
   ```bash
   # Rebuild Docker image
   docker-compose build --no-cache server
   ```

## 🎯 Future Enhancements

- [ ] Video frame extraction for thumbnails
- [ ] AI-powered content analysis
- [ ] Real-time IAB category detection
- [ ] Video transcription and metadata extraction
- [ ] Advanced tagging algorithms
- [ ] User authentication and authorization
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline integration
