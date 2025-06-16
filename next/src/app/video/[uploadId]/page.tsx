"use client";

import type React from "react";
import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowLeft,
  Play,
  Pause,
  Download,
  Clock,
  FileVideo,
  Music,
  ImageIcon,
  AlertCircle,
  CheckCircle,
  Loader2,
  Upload,
  Images,
  MessageSquare,
  Copy,
  Brain,
  Tag,
  FileText,
  Star,
  Mic,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";

interface VideoInfo {
  duration: number;
  size: number;
  bitrate: number;
  video?: {
    codec: string;
    width: number;
    height: number;
    fps: number;
  };
  audio?: {
    codec: string;
    sampleRate: string;
    channels: number;
  };
}

interface ProcessingResults {
  frames: {
    count: number;
    directory: string;
    extractionMethod: string;
  };
  audio: {
    path: string;
    size: number;
    format: string;
    bitrate: string;
    sampleRate: string;
  };
}

interface IABCategory {
  iab_code: string;
  iab_name: string;
  confidence: number;
  original_input: string;
  mapping_method: string;
}

interface AIAnalysis {
  visual_result?: any;
  audio_result?: any;
  analyzedAt: string;
  // New merged structure fields
  visual?: {
    analysis?: any;
    confidence?: number;
    tags?: any[];
    objects?: any[];
    scenes?: any[];
    text?: string | null;
    emotions?: any[];
    activities?: any[];
    locations?: any[];
    iab_categories?: IABCategory[];
  };
  audio?: {
    analysis?: any;
    confidence?: number;
    transcription?: string | null;
    sentiment?: any;
    language?: string | null;
    speakers?: number;
    emotions?: any[];
    topics?: any[];
    keywords?: any[];
    iab_categories?: IABCategory[];
  };
  combined?: {
    summary?: string;
    categories?: any[];
    sentiment?: any;
    confidence?: number;
    insights?: any[];
    coherence_score?: number;
    content_type?: string;
    iab_categories?: IABCategory[];
  };
  rawResponse?: any[];
}

interface FileInfo {
  id: string;
  filename: string;
  originalname: string;
  mimetype: string;
  size: number;
  uploadedAt: string;
  videoInfo?: VideoInfo;
  processing?: {
    status: string;
    startedAt?: string;
    completedAt?: string;
    results?: ProcessingResults;
  };
  aiAnalysis?: AIAnalysis;
}

interface Frame {
  filename: string;
  url: string;
  size?: number;
  dimensions?: {
    width: number;
    height: number;
  };
}

export default function VideoViewPage() {
  const params = useParams();
  const router = useRouter();
  const uploadId = params.uploadId as string;

  const [fileInfo, setFileInfo] = useState<FileInfo | null>(null);
  const [frames, setFrames] = useState<Frame[]>([]);
  const [processingStatus, setProcessingStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retryingAnalysis, setRetryingAnalysis] = useState(false);
  const [retryError, setRetryError] = useState<string | null>(null);
  const [transcription, setTranscription] = useState<any>(null);
  const [transcriptionLoading, setTranscriptionLoading] = useState(false);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3001";

  useEffect(() => {
    if (uploadId) {
      fetchVideoData();
      // Poll for processing status if still processing
      const statusInterval = setInterval(() => {
        // Only poll if status indicates ongoing work
        if (
          !processingStatus ||
          processingStatus.status === "processing" ||
          processingStatus.status === "ai-analysis" ||
          processingStatus.status === "uploading"
        ) {
          checkProcessingStatus();
        }
      }, 2000);
      return () => clearInterval(statusInterval);
    }
  }, [uploadId, processingStatus?.status]);

  const fetchVideoData = async () => {
    try {
      setLoading(true);
      
      // Fetch file info
      const fileResponse = await fetch(`${API_URL}/api/files/${uploadId}`);
      if (!fileResponse.ok) throw new Error("Video not found");
      
      const fileData = await fileResponse.json();
      setFileInfo(fileData);
      
      // Check processing status
      await checkProcessingStatus();

      // If processing is complete, fetch frames and transcription
      if (fileData.processing?.status === "completed") {
        await fetchFrames();
        await fetchTranscription();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load video");
    } finally {
      setLoading(false);
    }
  };

  const checkProcessingStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/files/${uploadId}/status`);
      if (response.ok) {
        const status = await response.json();
        console.log(
          "Processing status update:",
          status.status,
          status.aiAnalysis ? "with AI analysis" : "no AI analysis"
        );
        setProcessingStatus(status);
        
        // If AI analysis results are included in status, update file info immediately
        if (status.aiAnalysis && fileInfo) {
          console.log("Updating fileInfo with AI analysis from status");
          setFileInfo((prev) =>
            prev
              ? {
            ...prev,
                  aiAnalysis: status.aiAnalysis,
                }
              : prev
          );
        }
        
        // If processing completed, fetch frames and update file info
        if (status.status === "completed") {
          console.log("Processing completed, fetching frames and file info");
          if (frames.length === 0) {
            await fetchFrames();
          }
          // Fetch transcription if not already loaded
          if (!transcription) {
            await fetchTranscription();
          }
          // Also fetch updated file info to get AI analysis results
          await fetchFileInfo();
        }
      }
    } catch (err) {
      console.error("Failed to check processing status:", err);
    }
  };

  const fetchFrames = async () => {
    try {
      const response = await fetch(`${API_URL}/api/files/${uploadId}/frames`);
      if (response.ok) {
        const framesData = await response.json();

        const framesWithUrls = framesData.frames.map((frame: any) => ({
          ...frame,
          url: `${API_URL}/api/files/${uploadId}/frames/${frame.filename}`,
        }));
        setFrames(framesWithUrls);
      }
    } catch (err) {
      console.error("Failed to fetch frames:", err);
    }
  };

  const fetchFileInfo = async () => {
    try {
      const fileResponse = await fetch(`${API_URL}/api/files/${uploadId}`);
      if (fileResponse.ok) {
        const fileData = await fileResponse.json();
        console.log(
          "Fetched file info:",
          fileData.aiAnalysis ? "with AI analysis" : "no AI analysis"
        );
        console.log("AI Analysis data:", fileData.aiAnalysis);
        setFileInfo(fileData);
      }
    } catch (err) {
      console.error("Failed to fetch file info:", err);
    }
  };

  const fetchTranscription = async () => {
    try {
      setTranscriptionLoading(true);
      const response = await fetch(
        `${API_URL}/api/files/${uploadId}/transcription`
      );
      if (response.ok) {
        const transcriptionData = await response.json();
        console.log(
          "Fetched transcription:",
          transcriptionData.transcriptionAvailable
            ? "available"
            : "not available"
        );
        setTranscription(transcriptionData);
      }
    } catch (err) {
      console.error("Failed to fetch transcription:", err);
    } finally {
      setTranscriptionLoading(false);
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ["Bytes", "KB", "MB", "GB"];
    if (bytes === 0) return "0 Bytes";
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + " " + sizes[i];
  };

  const retryAIAnalysis = async () => {
    if (retryingAnalysis) return;

    setRetryingAnalysis(true);
    setRetryError(null); // Clear any previous retry errors

    try {
      const response = await fetch(
        `${API_URL}/api/files/${uploadId}/retry-analysis`,
        {
          method: "POST",
        }
      );
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to retry AI analysis");
      }
      
      // Reset to show progress again and continue polling
      await checkProcessingStatus();
    } catch (err) {
      console.error("Failed to retry AI analysis:", err);
      // Set retry error instead of global error
      setRetryError(
        err instanceof Error ? err.message : "Failed to retry AI analysis"
      );
    } finally {
      setRetryingAnalysis(false);
    }
  };

  const openFrameInNewTab = (frameUrl: string, filename: string) => {
    window.open(frameUrl, "_blank", "noopener,noreferrer");
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-purple-600" />
          <p className="text-slate-600">Loading video...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 flex items-center justify-center">
        <Card className="border-0 shadow-xl bg-white/70 backdrop-blur-sm max-w-md">
          <CardContent className="p-8 text-center">
            <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-slate-900 mb-2">
              Video Not Found
            </h2>
            <p className="text-slate-600 mb-4">{error}</p>
            <Button onClick={() => router.push("/")} variant="outline">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Upload
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 p-6">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-slate-200 bg-white/80 backdrop-blur-sm mb-6 -mx-6 px-6 py-4">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button variant="outline" onClick={() => router.push("/")}>
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back
              </Button>
              <div>
                <h1 className="text-xl font-bold text-slate-900">
                  {fileInfo?.originalname}
                </h1>
                <p className="text-sm text-slate-600">
                  Uploaded{" "}
                  {new Date(fileInfo?.uploadedAt || "").toLocaleString()}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Button variant="outline" onClick={() => router.push("/")}>
                <Upload className="w-4 h-4 md:mr-2" />
                <span className="hidden md:inline">Upload Another Video</span>
              </Button>
              <Button variant="outline" onClick={() => router.push("/gallery")}>
                <Images className="w-4 h-4 md:mr-2" />
                <span className="hidden md:inline">View Gallery</span>
              </Button>
              {processingStatus && (
                <Badge
                  variant={
                    processingStatus.status === "completed"
                      ? "default"
                      : processingStatus.status === "processing"
                      ? "secondary"
                      : processingStatus.status === "ai-analysis"
                      ? "secondary"
                      : processingStatus.status === "ai-analysis-error"
                      ? "destructive"
                      : processingStatus.status === "error"
                      ? "destructive"
                      : "outline"
                  }
                  className={
                    processingStatus.status === "completed"
                      ? "bg-green-100 text-green-700 border-green-200"
                      : processingStatus.status === "processing"
                      ? "bg-yellow-100 text-yellow-700 border-yellow-200"
                      : processingStatus.status === "ai-analysis"
                      ? "bg-purple-100 text-purple-700 border-purple-200"
                      : processingStatus.status === "ai-analysis-error"
                      ? "bg-red-100 text-red-700 border-red-200"
                      : ""
                  }
                >
                  {processingStatus.status === "completed" && (
                    <CheckCircle className="w-3 h-3 mr-1" />
                  )}
                  {(processingStatus.status === "processing" ||
                    processingStatus.status === "ai-analysis") && (
                    <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                  )}
                  {(processingStatus.status === "error" ||
                    processingStatus.status === "ai-analysis-error") && (
                    <AlertCircle className="w-3 h-3 mr-1" />
                  )}
                  {processingStatus.status === "completed"
                    ? "Completed"
                    : processingStatus.status === "processing"
                    ? "Processing..."
                    : processingStatus.status === "ai-analysis"
                    ? "AI Analysis..."
                    : processingStatus.status === "ai-analysis-error"
                    ? "AI Analysis Failed"
                    : processingStatus.status === "error"
                    ? "Error"
                    : processingStatus.status}
                </Badge>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto space-y-6">
        {/* Processing Status */}
        {(processingStatus?.status === "processing" ||
          processingStatus?.status === "ai-analysis") && (
          <Card className="border-0 shadow-xl bg-white/70 backdrop-blur-sm">
            <CardContent className="p-6">
              <div className="flex items-center space-x-4">
                <Loader2
                  className={`w-6 h-6 animate-spin ${
                    processingStatus.status === "ai-analysis"
                      ? "text-purple-600"
                      : "text-purple-600"
                  }`}
                />
                <div className="flex-1">
                  <h3 className="font-semibold text-slate-900">
                    {processingStatus.status === "ai-analysis"
                      ? "AI Analysis in Progress"
                      : "Processing Video"}
                  </h3>
                  <p className="text-sm text-slate-600">
                    {processingStatus.message}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Main Video Player */}
        <Card className="border-0 shadow-xl bg-white/70 backdrop-blur-sm overflow-hidden">
          <CardContent className="p-0">
            <div className="relative bg-black">
              <video
                className="w-full h-[400px] object-contain"
                controls
                poster="/placeholder.svg?height=400&width=800"
              >
                <source
                  src={`${API_URL}/api/files/${uploadId}/video`}
                  type={fileInfo?.mimetype}
                />
                Your browser does not support the video tag.
              </video>
                    </div>
            </CardContent>
          </Card>

                        {/* Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - AI Analysis Only */}
          <div className="lg:col-span-1 space-y-6">
            {/* AI Analysis Results - Main Feature */}
            <Card className="border-0 shadow-2xl bg-gradient-to-br from-purple-50 via-white to-indigo-50 backdrop-blur-sm ring-2 ring-purple-200/50 relative overflow-hidden p-0">
              {/* Decorative background element */}
              <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-purple-400/10 to-indigo-400/10 rounded-full blur-3xl"></div>
              <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-indigo-400/10 to-purple-400/10 rounded-full blur-2xl"></div>
              
              <CardHeader className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white relative rounded-t-xl p-4 m-0">
                <CardTitle className="flex items-center gap-3 text-xl">
                  <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
                    <Brain className="h-6 w-6" />
                  </div>
                  <div>
                    <div className="text-xl font-bold">AI Analysis Results</div>
                    <div className="text-purple-100 text-sm font-normal">Powered by Advanced AI</div>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent className="pb-4">
                {processingStatus?.status === "ai-analysis-error" ? (
                  <div className="space-y-3">
                    <div className="border border-orange-200 bg-orange-50 rounded-lg p-4">
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0">
                          <AlertCircle className="w-5 h-5 text-orange-600 mt-0.5" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <h4 className="text-sm font-medium text-orange-800 mb-1">
                            AI Analysis Not Available
                          </h4>
                          <p className="text-sm text-orange-700 mb-3">
                            Unable to generate content insights. Please try
                            again.
                          </p>
                  <Button 
                    onClick={retryAIAnalysis}
                    disabled={retryingAnalysis}
                            variant="outline"
                            size="sm"
                            className="border-orange-300 text-orange-700 hover:bg-orange-100 hover:text-orange-800"
                  >
                    {retryingAnalysis ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                Retrying
                      </>
                    ) : (
                      <>
                                <Brain className="w-4 h-4 mr-2" />
                                Retry
                      </>
                    )}
                  </Button>
                </div>
              </div>
                </div>
                    {retryError && (
                      <div className="border border-red-200 bg-red-50 rounded-lg p-3">
                        <div className="flex items-start gap-2">
                          <AlertCircle className="w-4 h-4 text-red-600 mt-0.5 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm text-red-700">
                              <strong>Retry failed:</strong> {retryError}
                            </p>
            </div>
                          <button
                            onClick={() => setRetryError(null)}
                            className="text-red-600 hover:text-red-800 text-xs"
                          >
                            ✕
                          </button>
                              </div>
                              </div>
                            )}
                                </div>
                ) : !fileInfo?.aiAnalysis &&
                  processingStatus?.status !== "completed" ? (
                  <div className="flex flex-col items-center justify-center py-16 text-center">
                    <div className="relative mb-6">
                      <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-indigo-500 rounded-full flex items-center justify-center">
                        <Loader2 className="w-10 h-10 animate-spin text-white" />
                              </div>
                      <div className="absolute -inset-2 bg-gradient-to-r from-purple-500 to-indigo-500 rounded-full blur opacity-20 animate-pulse"></div>
                                </div>
                    <h3 className="text-xl font-bold text-slate-900 mb-2">AI Analysis in Progress</h3>
                    <p className="text-slate-600 mb-4">Our advanced AI is analyzing your video content...</p>
                    <div className="flex items-center gap-2 text-sm text-purple-600">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                        <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                              </div>
                      <span className="font-medium">Processing</span>
                                </div>
                              </div>
                ) : (
                  <div className="space-y-4">
                    <div className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg p-4 border border-purple-100">
                      <h4 className="font-bold mb-3 flex items-center gap-2 text-base text-purple-900">
                        <div className="p-1.5 bg-purple-200 rounded-md">
                          <FileText className="h-4 w-4 text-purple-700" />
                          </div>
                        IAB Categories
                      </h4>
                      <div className="space-y-2">
                        {(() => {
                          // Get validated IAB categories from the new structure
                          const combinedIabCategories =
                            fileInfo?.aiAnalysis?.combined?.iab_categories ||
                            [];
                          const visualIabCategories =
                            fileInfo?.aiAnalysis?.visual?.iab_categories || [];
                          const audioIabCategories =
                            fileInfo?.aiAnalysis?.audio?.iab_categories || [];

                          // Use combined categories first, then fallback to individual ones
                          let categoriesToShow = combinedIabCategories;
                          if (categoriesToShow.length === 0) {
                            categoriesToShow = [
                              ...visualIabCategories,
                              ...audioIabCategories,
                            ];
                          }

                          // Fallback to legacy format if new structure not available
                          if (categoriesToShow.length === 0) {
                            const visualData =
                              fileInfo?.aiAnalysis?.visual_result ||
                              fileInfo?.aiAnalysis?.visual?.analysis;
                            const audioData =
                              fileInfo?.aiAnalysis?.audio_result ||
                              fileInfo?.aiAnalysis?.audio?.analysis;
                            const legacyCategories = [];

                            // Add visual categories (primary + secondary)
                            if (visualData?.iab_categories?.primary) {
                              legacyCategories.push({
                                iab_code: "Unknown",
                                iab_name: visualData.iab_categories.primary,
                                confidence:
                                  visualData.iab_categories.confidence_score ||
                                  0.9,
                                original_input:
                                  visualData.iab_categories.primary,
                                mapping_method: "legacy_format",
                              });
                            }
                            if (visualData?.iab_categories?.secondary) {
                              visualData.iab_categories.secondary.slice(0, 2).forEach((category: string) => {
                                legacyCategories.push({
                                  iab_code: "Unknown",
                                  iab_name: category,
                                  confidence: 0.7,
                                  original_input: category,
                                  mapping_method: "legacy_format",
                                });
                              });
                            }

                            // Add audio categories (primary + secondary)
                            if (audioData?.iab_categories?.primary) {
                              legacyCategories.push({
                                iab_code: "Unknown",
                                iab_name: audioData.iab_categories.primary,
                                confidence:
                                  audioData.iab_categories.confidence_score ||
                                  0.9,
                                original_input:
                                  audioData.iab_categories.primary,
                                mapping_method: "legacy_format",
                              });
                            }
                            if (audioData?.iab_categories?.secondary) {
                              audioData.iab_categories.secondary.slice(0, 2).forEach((category: string) => {
                                legacyCategories.push({
                                  iab_code: "Unknown",
                                  iab_name: category,
                                  confidence: 0.7,
                                  original_input: category,
                                  mapping_method: "legacy_format",
                                });
                              });
                            }
                            categoriesToShow = legacyCategories;
                          }

                          return categoriesToShow.length > 0 ? (
                            categoriesToShow.slice(0, 3).map((item, index) => (
                              <div
                                key={index}
                                className="flex items-center justify-between bg-white/80 rounded-xl p-4 border border-purple-100 hover:shadow-md transition-all"
                              >
                                <div className="flex-1">
                                  <div className="flex items-center gap-3">
                                    <span className="text-base font-semibold text-slate-800">
                                      {item.iab_name}
                                    </span>
                                    {item.iab_code &&
                                      item.iab_code !== "Unknown" && (
                                        <Badge
                                          variant="outline"
                                          className="text-xs bg-purple-100 text-purple-700 border-purple-300"
                                        >
                                          {item.iab_code}
                                  </Badge>
                                )}
                              </div>
                                </div>
                                <div className="flex items-center gap-3 ml-4">
                                  <div className="w-20 bg-purple-100 rounded-full h-3 overflow-hidden">
                                    <div
                                      className="bg-gradient-to-r from-purple-500 to-indigo-500 h-3 rounded-full transition-all duration-1000 ease-out"
                                      style={{
                                        width: `${
                                          (item.confidence || 0) * 100
                                        }%`,
                                      }}
                                    />
                              </div>
                                  <span className="text-sm font-bold text-purple-700 min-w-[3rem]">
                                    {Math.round((item.confidence || 0) * 100)}%
                                  </span>
                              </div>
                          </div>
                            ))
                          ) : (
                            <p className="text-sm text-slate-500 bg-white/60 rounded-lg p-4 text-center">
                              No IAB categories available
                            </p>
                        );
                    })()}
                      </div>
            </div>

                    <div className="bg-gradient-to-r from-indigo-50 to-blue-50 rounded-lg p-4 border border-indigo-100">
                      <h4 className="font-bold mb-3 flex items-center gap-2 text-base text-indigo-900">
                        <div className="p-1.5 bg-indigo-200 rounded-md">
                          <Tag className="h-4 w-4 text-indigo-700" />
                          </div>
                        Generated Tags
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {(() => {
                          // Get tags from processed structure first, then fallback to raw response
                          const processedVisual = fileInfo?.aiAnalysis?.visual;
                          const processedAudio = fileInfo?.aiAnalysis?.audio;
                          const rawVisual = fileInfo?.aiAnalysis?.visual_result;
                          const rawAudio = fileInfo?.aiAnalysis?.audio_result;

                          const allTags = [];

                          // Extract tags from raw webhook response
                          if (rawVisual?.tags) {
                            const visualTags = rawVisual.tags;
                            if (
                              typeof visualTags === "object" &&
                              !Array.isArray(visualTags)
                            ) {
                              // Extract from nested structure
                              [
                                "objects",
                                "activities",
                                "people",
                                "mood",
                                "brands_text",
                              ].forEach((category) => {
                                if (
                                  visualTags[category] &&
                                  Array.isArray(visualTags[category])
                                ) {
                                  allTags.push(
                                    ...visualTags[category].slice(0, 2)
                                  );
                                }
                              });
                            } else if (Array.isArray(visualTags)) {
                              allTags.push(...visualTags.slice(0, 5));
                            }
                          }

                          if (rawAudio?.tags) {
                            const audioTags = rawAudio.tags;
                            if (
                              typeof audioTags === "object" &&
                              !Array.isArray(audioTags)
                            ) {
                              // Extract from nested structure
                              ["keywords", "descriptive"].forEach(
                                (category) => {
                                  if (
                                    audioTags[category] &&
                                    Array.isArray(audioTags[category])
                                  ) {
                                    allTags.push(
                                      ...audioTags[category].slice(0, 3)
                                    );
                                  }
                                }
                              );
                            } else if (Array.isArray(audioTags)) {
                              allTags.push(...audioTags.slice(0, 5));
                            }
                          }

                          // Fallback to processed data if raw tags not available
                          if (allTags.length === 0) {
                            if (processedVisual?.tags)
                              allTags.push(...processedVisual.tags.slice(0, 5));
                            if (processedAudio?.keywords)
                              allTags.push(
                                ...processedAudio.keywords.slice(0, 5)
                              );
                          }

                          return allTags.length > 0 ? (
                            allTags.map((tag, index) => (
                              <Badge key={index} variant="secondary" className="bg-indigo-100 text-indigo-800 border-indigo-200 hover:bg-indigo-200 px-3 py-1 text-sm font-medium">
                                      {tag}
                                    </Badge>
                            ))
                          ) : (
                            <p className="text-sm text-slate-500 text-center bg-white/60 rounded-lg p-4">
                              No tags available
                            </p>
                          );
                        })()}
                                </div>
                              </div>

                    <div className="bg-gradient-to-r from-emerald-50 to-teal-50 rounded-lg p-4 border border-emerald-100">
                      <h4 className="font-bold mb-3 flex items-center gap-2 text-base text-emerald-900">
                        <div className="p-1.5 bg-emerald-200 rounded-md">
                          <MessageSquare className="h-4 w-4 text-emerald-700" />
                          </div>
                        Topics
                      </h4>
                      <div className="space-y-2">
                        {(() => {
                          // Get topics from processed structure first, then fallback to raw response
                          const processedVisual = fileInfo?.aiAnalysis?.visual;
                          const processedAudio = fileInfo?.aiAnalysis?.audio;
                          const rawVisual = fileInfo?.aiAnalysis?.visual_result;
                          const rawAudio = fileInfo?.aiAnalysis?.audio_result;

                          const topics = [];

                          // Add processed topics
                          if (processedAudio?.topics)
                            topics.push(...processedAudio.topics.slice(0, 4));
                          if (processedVisual?.scenes)
                            topics.push(...processedVisual.scenes.slice(0, 2));

                          // Fallback to raw response topics
                          if (topics.length === 0) {
                            if (rawVisual?.topics)
                              topics.push(...rawVisual.topics.slice(0, 4));
                            if (rawAudio?.topics)
                              topics.push(...rawAudio.topics.slice(0, 4));
                          }

                          return topics.length > 0 ? (
                            topics.map((topic, index) => (
                              <div
                                key={index}
                                className="flex items-center gap-3 bg-white/80 rounded-lg p-3 border border-emerald-100 hover:shadow-sm transition-all"
                              >
                                <div className="p-1 bg-emerald-100 rounded">
                                  <MessageSquare className="h-3 w-3 text-emerald-600" />
                        </div>
                                <span className="text-sm font-medium text-slate-800">{topic}</span>
                    </div>
                            ))
                          ) : (
                            <p className="text-sm text-slate-500 text-center bg-white/60 rounded-lg p-4">
                              No topics available
                            </p>
              );
            })()}
          </div>
                </div>

                    <div className="bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg p-4 border border-amber-100">
                      <h4 className="font-bold mb-3 flex items-center gap-2 text-base text-amber-900">
                        <div className="p-1.5 bg-amber-200 rounded-md">
                          <Brain className="h-4 w-4 text-amber-700" />
            </div>
                        Content Insights
                      </h4>
                      <div className="space-y-4">
                        <div>
                          <h5 className="text-sm font-bold text-amber-800 mb-2 flex items-center gap-2">
                            <FileText className="h-3 w-3" />
                            Content Description
                          </h5>
                          {(() => {
                            const processedVisual =
                              fileInfo?.aiAnalysis?.visual;
                            const processedAudio =
                              fileInfo?.aiAnalysis?.audio;
                            const rawVisual =
                              fileInfo?.aiAnalysis?.visual_result;
                            const rawAudio =
                              fileInfo?.aiAnalysis?.audio_result;

                            const visualDesc = processedVisual?.analysis || rawVisual?.summary;
                            const audioDesc = processedAudio?.analysis || rawAudio?.summary;

                            if (visualDesc && audioDesc) {
                              return (
                                <div className="space-y-4">
                                  <div className="bg-white/80 rounded-lg p-4 border border-amber-100">
                                    <h6 className="text-sm font-bold text-amber-700 mb-2 flex items-center gap-2">
                                      <ImageIcon className="h-3 w-3" />
                                      Visual Analysis
                                    </h6>
                                    <p className="text-sm text-slate-700 leading-relaxed">{visualDesc}</p>
                    </div>
                                  <div className="bg-white/80 rounded-lg p-4 border border-amber-100">
                                    <h6 className="text-sm font-bold text-amber-700 mb-2 flex items-center gap-2">
                                      <Music className="h-3 w-3" />
                                      Audio Analysis
                                    </h6>
                                    <p className="text-sm text-slate-700 leading-relaxed">{audioDesc}</p>
                  </div>
                      </div>
                              );
                            } else {
                              return (
                                <p className="text-sm text-slate-700 leading-relaxed bg-white/80 rounded-lg p-4 border border-amber-100">
                                  {visualDesc || audioDesc || "AI analysis provides comprehensive insights into video content, categorization, and key themes."}
                                </p>
                              );
                            }
                          })()}
                    </div>
                        <div>
                          <h5 className="text-sm font-bold text-amber-800 mb-2 flex items-center gap-2">
                            <Star className="h-3 w-3" />
                            Content Safety
                          </h5>
                          <div className="bg-white/80 rounded-lg p-3 border border-amber-100">
                            <Badge
                              variant="outline"
                              className="text-green-600 border-green-600 bg-green-50 px-4 py-2 text-sm font-semibold"
                            >
                              <Star className="h-4 w-4 mr-2" />
                              {(() => {
                                // Access raw webhook response data
                                const rawResponse = fileInfo?.aiAnalysis?.rawResponse;
                                let visualBrandSafe = null;
                                let audioBrandSafe = null;
                                
                                if (Array.isArray(rawResponse)) {
                                  rawResponse.forEach(item => {
                                    if (item.visual_result) {
                                      visualBrandSafe = item.visual_result.content_safety?.brand_safe || item.visual_result.advertising_suitability?.brand_safe;
                                    }
                                    if (item.audio_result) {
                                      audioBrandSafe = item.audio_result.content_safety?.brand_safe || item.audio_result.content_safety?.advertiser_friendly;
                                    }
                                  });
                                }
                                
                                if (visualBrandSafe === true || audioBrandSafe === true) {
                                  return "Brand Safe";
                                } else if (visualBrandSafe === false || audioBrandSafe === false) {
                                  return "Not Brand Safe";
                                } else {
                                  return "Analysis Pending";
                                }
                              })()}
                            </Badge>
                    </div>
                  </div>
                      </div>
                    </div>
                  </div>
                )}
                </CardContent>
              </Card>
            </div>

          {/* Right Column - Metadata and Frames */}
          <div className="lg:col-span-2 space-y-6">
            {/* Video Metadata */}
            <Card className="border-0 shadow-xl bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileVideo className="h-5 w-5" />
                  Video Metadata
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div>
                    <p className="text-sm font-medium text-slate-500">
                      Filename
                    </p>
                    <p
                      className="text-sm truncate"
                      title={fileInfo?.originalname || "Unknown"}
                    >
                      {fileInfo?.originalname || "Unknown"}
                    </p>
                </div>
                              <div>
                    <p className="text-sm font-medium text-slate-500">
                      Duration
                    </p>
                    <p className="text-sm flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {fileInfo?.videoInfo
                        ? formatDuration(fileInfo.videoInfo.duration)
                        : "Unknown"}
                    </p>
                </div>
                {fileInfo?.videoInfo?.video && (
                    <div>
                      <p className="text-sm font-medium text-slate-500">
                        Resolution
                      </p>
                      <p className="text-sm">
                        {fileInfo.videoInfo.video.width}×
                        {fileInfo.videoInfo.video.height}
                      </p>
                    </div>
                            )}
                  {fileInfo?.videoInfo?.video && (
                              <div>
                      <p className="text-sm font-medium text-slate-500">
                        Frame Rate
                      </p>
                      <p className="text-sm">
                        {Math.round(fileInfo.videoInfo.video.fps)} fps
                      </p>
                  </div>
                )}
                  <div>
                    <p className="text-sm font-medium text-slate-500">
                      File Size
                    </p>
                    <p className="text-sm">
                      {formatFileSize(fileInfo?.size || 0)}
                    </p>
                    </div>
                          <div>
                    <p className="text-sm font-medium text-slate-500">Format</p>
                    <p className="text-sm">
                      {fileInfo?.mimetype?.split("/")[1]?.toUpperCase() ||
                        "Unknown"}
                    </p>
                            </div>
                  {fileInfo?.videoInfo?.video && (
                    <div>
                      <p className="text-sm font-medium text-slate-500">
                        Codec
                      </p>
                      <p className="text-sm">
                        {fileInfo.videoInfo.video.codec}
                      </p>
                          </div>
                        )}
                  {fileInfo?.videoInfo?.bitrate && (
                          <div>
                      <p className="text-sm font-medium text-slate-500">
                        Bitrate
                      </p>
                      <p className="text-sm">
                        {Math.round(fileInfo.videoInfo.bitrate / 1000)}kbps
                      </p>
                          </div>
                        )}
                              <div>
                    <p className="text-sm font-medium text-slate-500">
                      Upload Date
                    </p>
                    <p className="text-sm">
                      {new Date(
                        fileInfo?.uploadedAt || ""
                      ).toLocaleDateString()}
                    </p>
          </div>
        </div>
                  </CardContent>
                </Card>

            {/* Extracted Frames */}
            <Card className="border-0 shadow-xl bg-white/70 backdrop-blur-sm">
                <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ImageIcon className="h-5 w-5" />
                  Extracted Frames
                  </CardTitle>
                </CardHeader>
                <CardContent>
                {frames.length > 0 ? (
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {frames.map((frame, index) => (
                      <div
                        key={frame.filename}
                        className="relative group cursor-pointer"
                        onClick={() =>
                          openFrameInNewTab(frame.url, frame.filename)
                        }
                      >
                          <img
                            src={frame.url}
                          alt={`Frame at ${index + 1}`}
                          className="w-full h-24 object-cover rounded-lg border-2 border-transparent group-hover:border-blue-500 transition-colors"
                          />
                        <div className="absolute bottom-1 left-1 bg-black bg-opacity-75 text-white text-xs px-1 rounded">
                          Frame {index + 1}
                        </div>
                        <Button
                          size="sm"
                          variant="secondary"
                          className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity h-6 w-6 p-0"
                        >
                          <Download className="h-3 w-3" />
                        </Button>
                      </div>
                    ))}
                  </div>
                ) : processingStatus?.status === "completed" ? (
                  <p className="text-sm text-slate-500">No frames extracted</p>
                ) : (
                  <div className="flex items-center gap-2 text-sm text-slate-500">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Extracting frames...
                        </div>
                      )}
                </CardContent>
              </Card>

            {/* Audio and Transcription */}
            <Card className="border-0 shadow-xl bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Mic className="h-5 w-5" />
                  Audio & Transcription
                </CardTitle>
              </CardHeader>
                <CardContent className="space-y-4">
                <div className="flex items-center gap-4">
                  <audio controls className="flex-1">
                    <source
                      src={`${API_URL}/api/files/${uploadId}/audio`}
                      type="audio/mpeg"
                    />
                    Your browser does not support the audio element.
                  </audio>
                  <Button variant="outline" size="sm">
                    <Download className="h-4 w-4 mr-2" />
                    Download
                  </Button>
                      </div>
                <Separator />
                <div>
                  <h4 className="font-medium mb-2">Transcription</h4>
                  <ScrollArea className="h-40 w-full border rounded-md p-3">
                    {transcriptionLoading ? (
                      <div className="flex items-center gap-2 text-sm text-slate-500">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Loading transcription...
                      </div>
                    ) : (
                      <p className="text-sm text-slate-700 leading-relaxed">
                        {transcription?.transcriptionAvailable
                          ? transcription.transcription.text ||
                            "No transcription text available"
                          : processingStatus?.status === "completed"
                          ? "Transcription not available"
                          : "Transcription processing..."}
                      </p>
                    )}
                  </ScrollArea>
                </div>
              </CardContent>
            </Card>
          </div>
    </div>
      </div>
    </div>
  );
} 
