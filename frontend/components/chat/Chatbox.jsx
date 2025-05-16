/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable import/no-unresolved */
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Send, Mic, Ban, EllipsisVertical } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import Markdown from "./Markdown";
import Spinner from "../ui/spinner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import useAudioRecorder from "@/hooks/useAudioRecorder";
import useVideoQueue from "@/hooks/useVideoQueue";

const MODEL_LLM = "qwen3:4b"
const BACKEND_URL_OLLAMA = "http://192.168.0.49:11434";
const BACKEND_URL_FASTAPI = "http://192.168.0.49:8000";

export default function Chatbox() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [taskQueue, setTaskQueue] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCalling, setIsCalling] = useState(false);
  const [currentThinkingContent, setCurrentThinkingContent] = useState("");
  const [currentResponse, setCurrentResponse] = useState("");
  const [isThinking, setIsThinking] = useState(false);

  const abortControllerRef = useRef(null);
  const chatBottomRef = useRef(null);
  const [generatedVideoUrl, setGeneratedVideoUrl] = useState("");

  const formatSeconds = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remaining = seconds % 60;
    return `${minutes}:${remaining < 10 ? "0" : ""}${remaining}`;
  };

    


  const {
    startRecording,
    stopRecording,
    recording,
    durationSeconds,
    visualizerData,
    sttMutation,
  } = useAudioRecorder();
  const { addVideo, updateVideo, isQueueEmpty, reset, videoRef, canvasRef } = useVideoQueue();

  // Scroll to bottom on new content
  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView();
    }
  }, [messages, currentThinkingContent, currentResponse]);

  // Send message to Ollama
  const sendMessageToOllama = async (userMessage) => {
    setIsLoading(true);
    setCurrentThinkingContent("");
    setCurrentResponse("");
    setIsThinking(true);

    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    const newMessages = [...messages, { role: "user", content: userMessage }];
    setMessages(newMessages);

    const allMessages = [
      {
        role: "system",
        content:
          "You are a helpful assistant. Always reply in English. Summarize content to be 100 words",
      },
      ...newMessages,
    ];

    try {
      const response = await fetch(`${BACKEND_URL_OLLAMA}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization:
            "Bearer AAAAC3NzaC1lZDI1NTE5AAAAIPTUQHHZp3yTa6WW01SWfjLhjGgjzapoSzdHv87m75mS",
        },
        body: JSON.stringify({
          model: MODEL_LLM,
          messages: allMessages,
          stream: true,
          options: { num_ctx: 4096 },
        }),
        signal,
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let thinkingContent = "";
      let responseContent = "";
      let inThinkMode = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            const contentChunk = data.message?.content || "";

            if (contentChunk.includes("<think>")) {
              inThinkMode = true;
              continue;
            }
            if (contentChunk.includes("</think>")) {
              inThinkMode = false;
              setIsThinking(false);
              continue;
            }

            if (inThinkMode) {
              thinkingContent += contentChunk;
              setCurrentThinkingContent(thinkingContent);
            } else {
              responseContent += contentChunk;
              setCurrentResponse(responseContent);
            }
          } catch (err) {
            console.error("Error parsing JSON:", err, line);
          }
        }
      }

      setMessages([
        ...newMessages,
        { role: "assistant", content: responseContent },
      ]);
      const messageId = Date.now();
      setTaskQueue((prev) => [
        ...prev,
        { id: messageId, text: responseContent },
      ]);
    } catch (err) {
      if (!(err.name === "AbortError")) {
        console.error("Error communicating with Ollama:", err);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              "Sorry, there was an error communicating with the AI service.",
          },
        ]);
      }
    } finally {
      setIsLoading(false);
      setIsThinking(false);
      abortControllerRef.current = null;
    }
  };

  // Form submit
  const handleSubmit = (e) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;
    sendMessageToOllama(input);
    setInput("");
  };

  // STT effect
  useEffect(() => {
    if (sttMutation.status === "success" && sttMutation.data) {
      console.log("Speech to text triggered");
      sendMessageToOllama(sttMutation.data.text);
      sttMutation.reset();
      stopRecording();
    }
  }, [sttMutation.status, sttMutation.data]);

  useEffect(() => {
    const processText = async (index, text) => {
      

      addVideo({ id: index, url: undefined });

      try {
        const response = await fetch(
          `${BACKEND_URL_FASTAPI}/v1/avatar-sse/text-to-video`,
          {
            method: "POST",
            headers: {
              Accept: "text/event-stream",
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              text,
              tts_speaker: "male",
              tts_length_scale: 1.0,
              lipsync_reversed: false,
              lipsync_starting_frame: 0,
              lipsync_enhance: true,
            }),
          }
        );
        if (!response.ok) throw new Error(`SSE error ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        const chunks = [];

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const raw = line.replace(/^data:\s*/, "");
            if (raw === "[DONE]") break;

            try {
              const payload = JSON.parse(raw);
              if (payload.status === "streaming" && payload.video_chunk_base64) {
                chunks.push(payload.video_chunk_base64.trim());
              }
            } catch (e) {
              console.error("Failed to parse SSE payload", e, raw);
            }
          }
        }

        // Decode each chunk and merge into one Uint8Array
        const byteArrays = chunks.map((chunk) => {
          const binary = atob(chunk);
          const len = binary.length;
          const arr = new Uint8Array(len);
          for (let i = 0; i < len; i++) {
            arr[i] = binary.charCodeAt(i);
          }
          return arr;
        });
        // concatenate
        const totalLength = byteArrays.reduce((sum, arr) => sum + arr.length, 0);
        const merged = new Uint8Array(totalLength);
        let offset = 0;
        for (const arr of byteArrays) {
          merged.set(arr, offset);
          offset += arr.length;
        }
        const blob = new Blob([merged.buffer], { type: 'video/mp4' });
        const url = URL.createObjectURL(blob);

        if (videoRef.current) {
          videoRef.current.src = url;
          videoRef.current.load();
          videoRef.current.play();
        }
        updateVideo(index, url, 0, false);
      } catch (err) {
        console.error("Lipsync SSE error:", err);
      } finally {
        setIsProcessing(false);
        setTaskQueue((prev) => prev.slice(1));
      }
    };

    if (taskQueue.length > 0 && !isProcessing) {
      setIsProcessing(true);
      processText(taskQueue[0].id, taskQueue[0].text);
    }
  }, [taskQueue, isProcessing, addVideo, updateVideo]);


  // Voice recording trigger
  useEffect(() => {
    if (
      isCalling &&
      !isLoading &&
      isQueueEmpty &&
      !recording &&
      !sttMutation.isSuccess &&
      !sttMutation.isPending
    ) {
      startRecording();
    }
  }, [
    isCalling,
    isLoading,
    isQueueEmpty,
    recording,
    sttMutation.isSuccess,
    sttMutation.isPending,
  ]);



  const handleStopChat = () => {
    if (abortControllerRef.current) abortControllerRef.current.abort();
    reset();
    setTaskQueue([]);
    setCurrentThinkingContent("");
    setCurrentResponse("");
    setIsThinking(false);
  };

  const handleStartRecording = () => {
    startRecording();
    setIsCalling(true);
  };

  const handleStopRecording = () => {
    stopRecording(false);
    setIsCalling(false);
    handleStopChat();
  };


  return (
    <>
      <ScrollArea className="h-72 grow p-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`mb-4 ${msg.role !== "user" ? "text-left" : "text-right"}`}
          >
            <div
              className={`inline-block p-2 rounded-lg ${msg.role !== "user"
                  ? "bg-muted text-muted-foreground"
                  : "bg-primary text-primary-foreground"
                }`}
            >
              <Markdown content={msg.content} />
            </div>
          </div>
        ))}

        {isThinking && currentThinkingContent && (
          <div className="mb-4 text-left">
            <div className="inline-block p-2 rounded-lg bg-gray-100 text-gray-500 dark:bg-gray-800 dark:text-gray-400">
              <h4 className="text-xs font-medium mb-1">Thinking...</h4>
              <Markdown content={currentThinkingContent} />
            </div>
          </div>
        )}

        {!isThinking &&
          currentResponse &&
          !messages.some(
            (m) => m.role === "assistant" && m.content === currentResponse
          ) && (
            <div className="mb-4 text-left">
              <div className="inline-block p-2 rounded-lg bg-muted text-muted-foreground">
                <Markdown content={currentResponse} />
              </div>
            </div>
          )}

        {sttMutation.isPending && (
          <div className="mb-4 text-right">
            <div className="inline-block p-2 rounded-lg bg-primary text-primary-foreground">
              <div className="inline-block size-8 border-4 border-t-transparent border-primary rounded-full animate-spin" />
            </div>
          </div>
        )}

        <div ref={chatBottomRef} />
      </ScrollArea>

      <div className="flex space-x-2">
        {isCalling && (isLoading || !isQueueEmpty || sttMutation.isPending) ? (
          <div
            className="flex flex-1 self-center items-center justify-center ml-2 mx-1 overflow-hidden h-6"
            dir="rtl"
          >
            <Spinner size={24} />
          </div>
        ) : (
          <div
            className="flex flex-1 self-center items-center justify-between ml-2 mx-1 overflow-hidden h-6"
            dir="rtl"
          >
            <span className="ml-2 text-sm text-gray-500">
              {formatSeconds(durationSeconds)}
            </span>
            <div className="flex items-center gap-0.5 h-6 w-full max-w-full overflow-hidden flex-wrap">
              {visualizerData
                .slice()
                .reverse()
                .map((rms, idx) => (
                  <div key={idx} className="flex items-center h-full">
                    <div
                      className={`w-[2px] shrink-0 ${recording
                          ? "bg-indigo-500 dark:bg-indigo-400"
                          : "bg-gray-500 dark:bg-gray-400"
                        } inline-block h-full`}
                      style={{
                        height: `${Math.min(100, Math.max(14, rms * 100))}%`,
                      }}
                    />
                  </div>
                ))}
            </div>
          </div>
        )}

        {!isCalling && !recording && (
          <>
            <Input
              type="text"
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              disabled={isLoading}
              className="grow"
            />
            {isLoading || !isQueueEmpty ? (
              <Button
                onClick={handleStopChat}
                variant="destructive"
                size="icon"
              >
                <Ban color="white" />
                <span className="sr-only">Stop Chat</span>
              </Button>
            ) : (
              <Button onClick={handleSubmit} size="icon" disabled={isLoading}>
                <Send className="size-4" />
                <span className="sr-only">Send</span>
              </Button>
            )}
          </>
        )}

        {isCalling ? (
          <Button
            size="icon"
            variant="destructive"
            onClick={handleStopRecording}
          >
            <Ban color="white" />
            <span className="sr-only">Stop recording</span>
          </Button>
        ) : (
          <Button
            size="icon"
            disabled={isLoading || !isQueueEmpty || sttMutation.isPending}
            onClick={handleStartRecording}
          >
            <Mic className="size-4" />
            <span className="sr-only">Voice input</span>
          </Button>
        )}


      </div>
    </>
  );
}
