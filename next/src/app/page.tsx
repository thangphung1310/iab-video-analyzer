"use client"

import type React from "react"

import { useState, useCallback, useEffect } from "react"
import { Upload, Video, CheckCircle, AlertCircle, Sparkles, Zap, Brain, Images } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { useRouter } from "next/navigation"

export default function VideoUploadPage() {
  const [isDragOver, setIsDragOver] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadComplete, setUploadComplete] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadId, setUploadId] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [serverStatus, setServerStatus] = useState<'checking' | 'online' | 'offline'>('checking')
  
  const router = useRouter()

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const validateFile = async (file: File): Promise<{ valid: boolean; error?: string }> => {
    // Check file size (100MB limit)
    const MAX_SIZE_MB = 100
    const fileSizeMB = file.size / (1024 * 1024)
    if (fileSizeMB > MAX_SIZE_MB) {
      return {
        valid: false,
        error: `File size (${fileSizeMB.toFixed(1)}MB) exceeds maximum limit of ${MAX_SIZE_MB}MB`
      }
    }

    // Check file type
    if (!file.type.startsWith('video/')) {
      return {
        valid: false,
        error: `Invalid file type. Please select a video file (MP4, MOV, AVI, WebM)`
      }
    }

    // Check video duration (15 minutes limit)
    const MAX_DURATION_MINUTES = 15
    try {
      const duration = await getVideoDuration(file)
      const durationMinutes = duration / 60
      
      if (durationMinutes > MAX_DURATION_MINUTES) {
        return {
          valid: false,
          error: `Video duration (${durationMinutes.toFixed(1)} minutes) exceeds maximum limit of ${MAX_DURATION_MINUTES} minutes`
        }
      }
    } catch (error) {
      // Silently continue if duration check fails
    }

    return { valid: true }
  }

  const getVideoDuration = (file: File): Promise<number> => {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video')
      video.preload = 'metadata'
      
      video.onloadedmetadata = () => {
        window.URL.revokeObjectURL(video.src)
        resolve(video.duration)
      }
      
      video.onerror = () => {
        window.URL.revokeObjectURL(video.src)
        reject(new Error('Could not load video metadata'))
      }
      
      video.src = URL.createObjectURL(file)
    })
  }

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    const videoFile = files.find((file) => file.type.startsWith("video/")) || files[0]

    if (videoFile) {
      const validation = await validateFile(videoFile)
      if (validation.valid) {
      setSelectedFile(videoFile)
        setUploadError(null)
      } else {
        setUploadError(validation.error || 'Invalid file')
        setSelectedFile(null)
      }
    }
  }, [])

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const validation = await validateFile(file)
      if (validation.valid) {
      setSelectedFile(file)
        setUploadError(null)
      } else {
        setUploadError(validation.error || 'Invalid file')
        setSelectedFile(null)
      }
    }
  }, [])

  const uploadVideo = async (file: File) => {
    setIsUploading(true)
    setUploadProgress(0)
    setUploadComplete(false)
    setUploadError(null)
    setUploadId(null)

    let progressInterval: NodeJS.Timeout | null = null

    try {
      const formData = new FormData()
      formData.append('video', file)

      progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            return prev
          }
          return prev + Math.random() * 8 + 2
        })
      }, 150)

      const response = await fetch('http://localhost:3001/api/upload', {
        method: 'POST',
        body: formData,
        mode: 'cors',
      })

      if (!response.ok) {
        if (progressInterval) clearInterval(progressInterval)
        let errorMessage = 'Upload failed'
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorData.error || errorData.message || errorMessage
        } catch (e) {
          errorMessage = `Server error: ${response.status} ${response.statusText}`
        }
        throw new Error(errorMessage)
      }

      const result = await response.json()
      setUploadId(result.uploadId)

      if (progressInterval) clearInterval(progressInterval)
        setUploadProgress(100)
        setIsUploading(false)
        setUploadComplete(true)
      
      router.push(`/video/${result.uploadId}`)

    } catch (error) {
      if (progressInterval) clearInterval(progressInterval)
      
      let errorMessage = 'Upload failed'
      
      if (error instanceof TypeError && error.message === 'Failed to fetch') {
        errorMessage = 'Cannot connect to server. Make sure the server is running on port 3001.'
      } else if (error instanceof Error) {
        errorMessage = error.message
      } else {
        errorMessage = `Unknown error: ${String(error)}`
      }
      
      setUploadError(errorMessage)
      setIsUploading(false)
      setUploadProgress(0)
    }
  }

  const resetUpload = () => {
    setSelectedFile(null)
    setUploadProgress(0)
    setIsUploading(false)
    setUploadComplete(false)
    setUploadId(null)
    setUploadError(null)
  }

  // Check server status
  const checkServerStatus = async () => {
    try {
      const response = await fetch('http://localhost:3001/api/health', {
        method: 'GET',
        mode: 'cors',
      })
      setServerStatus(response.ok ? 'online' : 'offline')
    } catch (error) {
      setServerStatus('offline')
    }
  }

  // Check server status on component mount
  useEffect(() => {
    checkServerStatus()
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-indigo-600 rounded-xl flex items-center justify-center">
                <Video className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">Video Tagging</h1>
                <p className="text-sm text-slate-600">AI-Powered IAB Category Detection</p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Button variant="outline" onClick={() => router.push('/gallery')}>
                <Images className="w-4 h-4 mr-2" />
                Gallery
              </Button>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${serverStatus === 'online' ? 'bg-green-500' : serverStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'}`} />
                <span className="text-sm text-slate-600 capitalize">{serverStatus}</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <div className="inline-flex items-center space-x-2 bg-purple-100 text-purple-700 px-4 py-2 rounded-full text-sm font-medium mb-6">
            <Sparkles className="w-4 h-4" />
            <span>AI-Powered Video Analysis</span>
          </div>
          <h1 className="text-4xl font-bold text-slate-900 mb-4">Upload Your Video</h1>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto">
            Transform your videos with cutting-edge AI technology. Get insights, summaries, and intelligent analysis in
            seconds.
          </p>
        </div>

        <Card className="border-0 shadow-xl bg-white/70 backdrop-blur-sm">
          <CardContent className="p-8">
            {serverStatus === 'offline' && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2 text-red-700">
                    <AlertCircle className="w-5 h-5" />
                    <span className="font-medium">Server Offline</span>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={checkServerStatus}
                    className="text-red-700 border-red-300 hover:bg-red-100"
                  >
                    Retry
                  </Button>
                </div>
                <p className="text-sm text-red-600 mt-1">
                  Cannot connect to the upload server. Please make sure the server is running.
                </p>
              </div>
            )}

            {!selectedFile ? (
              <div
                className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 ${
                  serverStatus === 'offline' 
                    ? "border-slate-200 bg-slate-50 opacity-50 cursor-not-allowed"
                    : isDragOver
                      ? "border-purple-400 bg-purple-50"
                      : "border-slate-300 hover:border-slate-400 hover:bg-slate-50"
                }`}
                onDragOver={serverStatus === 'online' ? handleDragOver : undefined}
                onDragLeave={serverStatus === 'online' ? handleDragLeave : undefined}
                onDrop={serverStatus === 'online' ? handleDrop : undefined}
              >
                <div className="flex flex-col items-center space-y-6">
                  <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-indigo-500 rounded-2xl flex items-center justify-center">
                    <Upload className="w-8 h-8 text-white" />
                  </div>

                  <div className="space-y-2">
                    <h3 className="text-xl font-semibold text-slate-900">Drop your video here</h3>
                    <p className="text-slate-600">or click to browse your files</p>
                  </div>

                  <div className="space-y-3">
                  <div className="flex flex-wrap justify-center gap-2 text-sm text-slate-500">
                    <Badge variant="outline">MP4</Badge>
                    <Badge variant="outline">MOV</Badge>
                    <Badge variant="outline">AVI</Badge>
                    <Badge variant="outline">WebM</Badge>
                    </div>
                    
                    <div className="text-xs text-slate-500 text-center space-y-1">
                      <p>📏 Maximum: 100MB file size</p>
                      <p>⏱️ Duration: up to 15 minutes</p>
                    </div>
                  </div>

                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleFileSelect}
                    disabled={serverStatus !== 'online'}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                  />
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="flex items-center space-x-4 p-4 bg-slate-50 rounded-xl">
                  <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                    <Video className="w-6 h-6 text-purple-600" />
                  </div>
                  <div className="flex-1">
                    <h4 className="font-medium text-slate-900">{selectedFile.name}</h4>
                    <p className="text-sm text-slate-600">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                  </div>
                  {uploadComplete ? (
                    <CheckCircle className="w-6 h-6 text-green-500" />
                  ) : isUploading ? (
                    <div className="w-6 h-6 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <AlertCircle className="w-6 h-6 text-slate-400" />
                  )}
                </div>

                {isUploading && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-600">Uploading...</span>
                      <span className="text-slate-900 font-medium">{Math.round(uploadProgress)}%</span>
                    </div>
                    <Progress value={uploadProgress} className="h-2" />
                  </div>
                )}

                {uploadComplete && (
                  <div className="p-4 bg-green-50 border border-green-200 rounded-xl">
                    <div className="flex items-center space-x-2 text-green-700">
                      <CheckCircle className="w-5 h-5" />
                      <span className="font-medium">Upload successful!</span>
                    </div>
                    <p className="text-sm text-green-600 mt-1">
                      Your video is being processed.
                    </p>
                    {uploadId && (
                      <p className="text-xs text-green-500 mt-1 font-mono">
                        Upload ID: {uploadId}
                      </p>
                    )}
                  </div>
                )}

                {uploadError && (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-xl">
                    <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2 text-red-700">
                      <AlertCircle className="w-5 h-5" />
                      <span className="font-medium">Upload failed</span>
                      </div>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={async () => {
                          try {
                            const response = await fetch('http://localhost:3001/api/health', {
                              method: 'GET',
                              mode: 'cors',
                            })
                            const healthData = await response.json()
                            console.log('Health check result:', healthData)
                            alert(`Server Status: ${healthData.status}\nCheck console for detailed diagnostics`)
                          } catch (error) {
                            alert('Cannot connect to server for diagnostics')
                          }
                        }}
                        className="text-red-700 border-red-300 hover:bg-red-100"
                      >
                        Diagnose
                      </Button>
                    </div>
                    <p className="text-sm text-red-600 mt-1">
                      {uploadError}
                    </p>
                    <details className="mt-2">
                      <summary className="text-xs text-red-500 cursor-pointer">Show troubleshooting tips</summary>
                      <div className="mt-1 text-xs text-red-600 space-y-1">
                        <p>• Check that the server is online</p>
                        <p>• Ensure your video file is under 100MB and less than 15 minutes</p>
                        <p>• Supported formats: MP4, MOV, AVI, WebM</p>
                        <p>• Try refreshing the page and uploading again</p>
                        <p>• Click "Diagnose" for detailed server status</p>
                      </div>
                    </details>
                  </div>
                )}

                <div className="flex space-x-3">
                  {uploadComplete ? (
                    <>
                      <Button className="flex-1 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700" onClick={() => router.push(`/video/${uploadId}`)}>
                        View Analysis
                      </Button>
                      <Button variant="outline" onClick={resetUpload}>
                        Upload Another
                      </Button>
                    </>
                  ) : isUploading ? (
                    <Button variant="outline" onClick={resetUpload} className="w-full">
                      Cancel Upload
                    </Button>
                  ) : (
                    <>
                      <Button 
                        className="flex-1 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700" 
                        onClick={() => uploadVideo(selectedFile!)}
                        disabled={!selectedFile}
                      >
                        <Brain className="w-4 h-4 mr-2" />
                        Analyze this video
                      </Button>
                      <Button variant="outline" onClick={resetUpload}>
                        Choose another video
                      </Button>
                    </>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-6 mt-12">
          <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
            <CardContent className="p-6 text-center">
              <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                <Brain className="w-6 h-6 text-purple-600" />
              </div>
              <h3 className="font-semibold text-slate-900 mb-2">IAB Category Detection</h3>
              <p className="text-sm text-slate-600">
                Automatically classify your video content using IAB standard categories for precise advertising targeting and content organization.
              </p>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
            <CardContent className="p-6 text-center">
              <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                <Zap className="w-6 h-6 text-green-600" />
              </div>
              <h3 className="font-semibold text-slate-900 mb-2">Smart Tagging</h3>
              <p className="text-sm text-slate-600">
                Extract relevant keywords, objects, scenes, and contextual tags from your video content with AI-powered analysis.
              </p>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-lg bg-white/70 backdrop-blur-sm">
            <CardContent className="p-6 text-center">
              <div className="w-12 h-12 bg-indigo-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                <Sparkles className="w-6 h-6 text-indigo-600" />
              </div>
              <h3 className="font-semibold text-slate-900 mb-2">Compliance Ready</h3>
              <p className="text-sm text-slate-600">
                Ensure your content meets advertising standards and brand safety requirements with automated compliance scoring.
              </p>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
