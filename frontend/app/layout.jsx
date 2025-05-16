/* eslint-disable import/no-unresolved */
// app/layout.jsx

import localFont from "next/font/local";

import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import Providers from "@/providers";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata = {
  title: "Digital Avatar with OpenVINO",
  description: "Create a digital avatar with OpenVINO and Next.js, powered by Local CPU, NPU and GPU.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <Providers>
          {children}
          <Toaster richColors />
        </Providers>
      </body>
    </html>
  );
}
