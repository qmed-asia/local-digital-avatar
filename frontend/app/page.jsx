/* eslint-disable @typescript-eslint/no-unused-expressions */
/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable import/no-unresolved */
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

"use client";
import { useEffect, useRef, useState } from "react";

import Chat from "@/components/chat";
import useVideoQueue from "@/hooks/useVideoQueue";

export default function Component() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
const [generatedVideoUrl, setGeneratedVideoUrl] = useState("/assets/video.mp4");

  const { handleVideoLoaded, updateRefs } = useVideoQueue();

  // Register DOM nodes with the shared context
  useEffect(() => {
    if (videoRef.current && canvasRef.current) {
      updateRefs(videoRef.current, canvasRef.current);
    }
  }, [updateRefs]);

  // Whenever the parent provides a new URL, load & play it
  useEffect(() => {
    // cant play video if not user interaction
    console.log("generatedVideoUrl", generatedVideoUrl);
    const video = videoRef.current
    if (generatedVideoUrl && video) {
      video.src = generatedVideoUrl
      video.load()
      video.play()
      video.playbackRate = 1.0
      
    }
  }, [generatedVideoUrl])

  return (
    <div className="h-screen bg-gradient-to-b from-primary/20 to-background">
      <div className="h-full grid grid-cols-1 md:grid-cols-12">
        <div className="md:col-span-4 relative overflow-hidden">
          <div className="w-screen h-screen relative">
            <video
              ref={videoRef}
              className="absolute inset-0 size-full object-contain object-left-top"
              poster="/assets/image.png"
              onLoadedData={handleVideoLoaded}
              // playsInline 
              loop={false}        
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 size-full object-contain object-left-top"
            />
          </div>
        </div>
        <div
          className={`md:col-span-8 flex flex-col bg-background rounded-t-3xl md:rounded-none shadow-lg h-full transition-all duration-300 ease-in-out`}
        >
          <Chat setGeneratedVideoUrl={setGeneratedVideoUrl} />
        </div>
      </div>
    </div>
  );
}
