"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { ArrowLeft, Play, Clock, FileVideo, Eye, AlertCircle, Loader2, Trash2, Tag, Hash } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface VideoFile {
  id: string
  filename: string
  originalname: string
  mimetype: string
  size: number
  uploadedAt: string
  videoInfo?: {
    duration: number
    video?: {
      width: number
      height: number
    }
  }
  processing?: {
    status: string
    results?: {
      frames: {
        count: number
      }
    }
  }
  aiAnalysis?: {
    visual?: {
      tags?: string[]
      iab_categories?: Array<{
        iab_code: string
        iab_name: string
        confidence: number
      }>
    }
    audio?: {
      keywords?: string[]
      topics?: string[]
      iab_categories?: Array<{
        iab_code: string
        iab_name: string
        confidence: number
      }>
    }
    combined?: {
      categories?: string[]
      iab_categories?: Array<{
        iab_code: string
        iab_name: string
        confidence: number
      }>
    }
    visual_result?: any
    audio_result?: any
    rawResponse?: any[]
  }
}

export default function VideoGalleryPage() {
  const [videos, setVideos] = useState<VideoFile[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const router = useRouter()

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001'

  useEffect(() => {
    fetchVideos()
  }, [])

  const fetchVideos = async () => {
    try {
      setLoading(true)
      const response = await fetch(`${API_URL}/api/files`)
      
      if (!response.ok) {
        throw new Error('Failed to fetch videos')
      }
      
      const videosData = await response.json()
      setVideos(videosData.reverse()) // Show newest first
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load videos')
    } finally {
      setLoading(false)
    }
  }

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    if (bytes === 0) return '0 Bytes'
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i]
  }

  const handleDelete = async (videoId: string) => {
    if (!confirm("Are you sure you want to delete this video? This cannot be undone.")) return;
    try {
      const response = await fetch(`${API_URL}/api/files/${videoId}`, { method: 'DELETE' });
      if (!response.ok) throw new Error('Failed to delete video');
      setVideos((prev) => prev.filter((v) => v.id !== videoId));
    } catch (err) {
      alert('Delete failed: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  }

  const getIABCategories = (video: VideoFile) => {
    const aiAnalysis = video.aiAnalysis;
    if (!aiAnalysis) return [];

    // Try combined categories first
    const combinedCategories = aiAnalysis.combined?.iab_categories || [];
    if (combinedCategories.length > 0) {
      return combinedCategories.slice(0, 2); // Show top 2
    }

    // Fallback to individual categories
    const visualCategories = aiAnalysis.visual?.iab_categories || [];
    const audioCategories = aiAnalysis.audio?.iab_categories || [];
    return [...visualCategories, ...audioCategories].slice(0, 2);
  }

  const getTags = (video: VideoFile) => {
    const aiAnalysis = video.aiAnalysis;
    if (!aiAnalysis) return [];

    const tags = [];
    
    // Extract from raw webhook response first
    if (aiAnalysis.rawResponse) {
      for (const item of aiAnalysis.rawResponse) {
        if (item.visual_result?.tags) {
          const visualTags = item.visual_result.tags;
          if (typeof visualTags === 'object' && !Array.isArray(visualTags)) {
            ['objects', 'activities', 'mood'].forEach(category => {
              if (visualTags[category] && Array.isArray(visualTags[category])) {
                tags.push(...visualTags[category].slice(0, 2));
              }
            });
          }
        }
        if (item.audio_result?.tags) {
          const audioTags = item.audio_result.tags;
          if (typeof audioTags === 'object' && !Array.isArray(audioTags)) {
            if (audioTags.keywords && Array.isArray(audioTags.keywords)) {
              tags.push(...audioTags.keywords.slice(0, 2));
            }
          }
        }
      }
    }

    // Fallback to processed data
    if (tags.length === 0) {
      if (aiAnalysis.visual?.tags) tags.push(...aiAnalysis.visual.tags.slice(0, 3));
      if (aiAnalysis.audio?.keywords) tags.push(...aiAnalysis.audio.keywords.slice(0, 3));
      if (aiAnalysis.audio?.topics) tags.push(...aiAnalysis.audio.topics.slice(0, 2));
    }

    return [...new Set(tags)].slice(0, 4); // Remove duplicates and limit to 4
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-purple-600" />
          <p className="text-slate-600">Loading videos...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 flex items-center justify-center">
        <Card className="border-0 shadow-xl bg-white/70 backdrop-blur-sm max-w-md">
          <CardContent className="p-8 text-center">
            <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-slate-900 mb-2">Failed to Load</h2>
            <p className="text-slate-600 mb-4">{error}</p>
            <Button onClick={fetchVideos} variant="outline">
              Try Again
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button variant="outline" onClick={() => router.push('/')}>
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Upload
              </Button>
              <div>
                <h1 className="text-2xl font-bold text-slate-900">Video Gallery</h1>
                <p className="text-sm text-slate-600">{videos.length} video{videos.length !== 1 ? 's' : ''} uploaded</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {videos.length === 0 ? (
          <Card className="border-0 shadow-xl bg-white/70 backdrop-blur-sm">
            <CardContent className="p-12 text-center">
              <FileVideo className="w-16 h-16 text-slate-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-slate-900 mb-2">No Videos Yet</h3>
              <p className="text-slate-600 mb-6">Upload your first video to get started with AI-powered analysis.</p>
              <Button onClick={() => router.push('/')} className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700">
                Upload Video
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {videos.map((video) => {
              const iabCategories = getIABCategories(video);
              const tags = getTags(video);
              const hasAnalysis = iabCategories.length > 0 || tags.length > 0;
              
              let cardClassName = "border-0 shadow-lg backdrop-blur-sm hover:shadow-xl transition-all group cursor-pointer ";
              if (hasAnalysis) {
                cardClassName += "bg-green-50/80 border-t-4 border-t-green-500"; // Green tint for analyzed videos
              } else if (video.processing?.status === 'completed') {
                cardClassName += "bg-purple-50/80 border-t-4 border-t-purple-400"; // Purple tint for completed but not analyzed
              } else {
                cardClassName += "bg-white/70"; // Default for processing/uploaded
              }
              
              return (
              <Card key={video.id} className={cardClassName}>
                <div onClick={() => router.push(`/video/${video.id}`)}>
                  <CardHeader className="p-4 flex flex-row items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="aspect-video bg-black rounded-lg overflow-hidden relative mb-3">
                        {/* Video thumbnail - using first frame if available */}
                        {video.processing?.status === 'completed' && video.processing.results?.frames?.count ? (
                          <img
                            src={`${API_URL}/api/files/${video.id}/frames/frame_001.jpg`}
                            alt={video.originalname}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              // Fallback to video element if frame not available
                              const target = e.target as HTMLImageElement
                              target.style.display = 'none'
                              const video = target.nextElementSibling as HTMLVideoElement
                              if (video) video.style.display = 'block'
                            }}
                          />
                        ) : null}
                        
                        <video
                          className="w-full h-full object-cover"
                          muted
                          style={{ display: video.processing?.status === 'completed' && video.processing.results?.frames?.count ? 'none' : 'block' }}
                        >
                          <source src={`${API_URL}/api/files/${video.id}/video`} type={video.mimetype} />
                        </video>
                        
                        {/* Play overlay */}
                        <div className="absolute inset-0 bg-opacity-0 group-hover:bg-opacity-30 transition-all flex items-center justify-center">
                          <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                            <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center shadow-lg">
                              <Eye className="w-6 h-6 text-slate-900" />
                            </div>
                          </div>
                        </div>
                        
                        {/* Duration badge */}
                        {video.videoInfo?.duration && (
                          <div className="absolute bottom-2 right-2">
                            <Badge variant="secondary" className="text-xs bg-black/70 text-white border-0">
                              <Clock className="w-3 h-3 mr-1" />
                              {formatDuration(video.videoInfo.duration)}
                            </Badge>
                          </div>
                        )}
                        
                        {/* Processing status */}
                        <div className="absolute top-2 left-2">
                          <Badge variant={
                            video.processing?.status === 'completed' ? 'default' :
                            video.processing?.status === 'processing' ? 'secondary' :
                            video.processing?.status === 'error' ? 'destructive' : 'outline'
                          } className={
                            video.processing?.status === 'completed' ? 'bg-green-100 text-green-700 border-green-200' :
                            video.processing?.status === 'processing' ? 'bg-yellow-100 text-yellow-700 border-yellow-200' :
                            'bg-black/70 text-white border-0'
                          }>
                            {video.processing?.status === 'completed' ? 'Processed' :
                             video.processing?.status === 'processing' ? 'Processing...' :
                             video.processing?.status === 'error' ? 'Error' :
                             'Uploaded'}
                          </Badge>
                        </div>
                      </div>
                      
                      <CardTitle className="text-sm font-medium text-slate-900 truncate" title={video.originalname}>
                        {video.originalname}
                      </CardTitle>
                    </div>
                    <Button size="icon" variant="ghost" className="text-red-500 hover:bg-red-100 ml-2" onClick={(e) => { e.stopPropagation(); handleDelete(video.id); }} title="Delete video">
                      <Trash2 className="w-5 h-5" />
                    </Button>
                  </CardHeader>
                  
                  <CardContent className="p-4 pt-0">
                                        {/* IAB Categories and Tags */}
                    {(() => {
                      const iabCategories = getIABCategories(video);
                      const tags = getTags(video);
                      const hasAnalysis = iabCategories.length > 0 || tags.length > 0;
                      
                      if (!hasAnalysis && video.processing?.status === 'completed') {
                        return (
                          <div className="mb-3 p-2 bg-slate-50 rounded border border-dashed border-slate-200">
                            <div className="flex items-center gap-1 text-slate-500">
                              <AlertCircle className="w-3 h-3" />
                              <span className="text-xs">No AI analysis available</span>
                            </div>
                          </div>
                        );
                      }
                      
                      return (
                        <>
                          {iabCategories.length > 0 && (
                            <div className="mb-3">
                              <div className="flex items-center gap-1 mb-2">
                                <Hash className="w-3 h-3 text-slate-500" />
                                <span className="text-xs font-medium text-slate-600">Categories</span>
                              </div>
                              <div className="flex flex-wrap gap-1">
                                {iabCategories.map((category, index) => (
                                  <Badge key={index} variant="outline" className="text-xs px-2 py-0.5">
                                    {category.iab_name}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}

                          {tags.length > 0 && (
                            <div className="mb-3">
                              <div className="flex items-center gap-1 mb-2">
                                <Tag className="w-3 h-3 text-slate-500" />
                                <span className="text-xs font-medium text-slate-600">Tags</span>
                              </div>
                              <div className="flex flex-wrap gap-1">
                                {tags.map((tag, index) => (
                                  <Badge key={index} variant="secondary" className="text-xs px-2 py-0.5">
                                    {tag}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}
                        </>
                      );
                    })()}

                    <div className="space-y-2 text-xs text-slate-600">
                      <div className="flex justify-between">
                        <span>Size:</span>
                        <span>{formatFileSize(video.size)}</span>
                      </div>
                      
                      {video.videoInfo?.video && (
                        <div className="flex justify-between">
                          <span>Resolution:</span>
                          <span>{video.videoInfo.video.width}×{video.videoInfo.video.height}</span>
                        </div>
                      )}
                      
                      <div className="flex justify-between">
                        <span>Uploaded:</span>
                        <span>{new Date(video.uploadedAt).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </CardContent>
                </div>
              </Card>
              );
            })}
          </div>
        )}
      </main>
    </div>
  )
} 
