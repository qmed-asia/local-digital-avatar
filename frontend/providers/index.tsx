/* eslint-disable import/no-unresolved */
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

"use client"

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { VideoQueueProvider } from "@/context/VideoQueueContextInstant";

// import { VideoQueueProvider } from "@/context/VideoQueueContext";

export default function Providers({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    const queryClient = new QueryClient()
    return (
        <QueryClientProvider client={queryClient}>
                <VideoQueueProvider>
                    {children}
                </VideoQueueProvider>
        </QueryClientProvider>
    )
}