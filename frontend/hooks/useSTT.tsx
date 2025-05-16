// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

"use client"

import { useMutation, UseMutationResult } from "@tanstack/react-query";

const STT_API_URL = "http://192.168.0.49:8000/v1/stt/transcriptions";

export function useGetSTT(): UseMutationResult<
  Record<string, any>,
  Error,
  { file: File; language: string }
> {
  return useMutation({
    mutationFn: async ({ file, language }) => {
        console.log(file)
        console.log(language)
      const formData = new FormData();
      formData.append("file", file);
      formData.append("language", language);

      const response = await fetch(STT_API_URL, {
        method: "POST",
        headers: {
          // this matches -H 'accept: application/json'
          accept: "application/json",
        },
        body: formData, // automatically sets Content-Type: multipart/form-data; boundary=...
      });

      if (!response.ok) {
        throw new Error(`STT API error: ${response.status} ${response.statusText}`);
      }

      return response.json();
    },
  });
}
