"use client";

import { useState, useEffect } from "react";
import axios from "axios";
import { useRouter } from "next/navigation";
import Header from "@/components/Header";
import UploadZone from "@/components/UploadZone";

export default function Upload() {
  const router = useRouter();
  const [status, setStatus] = useState<"idle" | "uploading" | "analyzing" | "success" | "error">("idle");
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [error, setError] = useState<string | undefined>();

  useEffect(() => {
    if (status === "success") {
      const timer = setTimeout(() => {
        router.push("/result");
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [status, router]);

  const handleFileSelect = async (file: File) => {
    try {
      setStatus("uploading");
      setError(undefined);

      // Convert file to base64 for preview in results page
      const reader = new FileReader();
      reader.onloadend = () => {
        sessionStorage.setItem("uploadedImage", reader.result as string);
      };
      reader.readAsDataURL(file);

      const formData = new FormData();
      formData.append("image", file);  // Changed from "file" to "image" to match Flask backend

      // Brief upload delay for UX
      await new Promise(resolve => setTimeout(resolve, 500));

      setStatus("analyzing");

      try {
        // Send to Flask backend /predict endpoint
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
        const response = await axios.post(`${apiUrl}/predict`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });

        console.log("Backend response:", response.data);

        // Check if prediction was successful
        if (response.data.success && response.data.prediction) {
          const result = {
            prediction: response.data.prediction.class,
            confidence: response.data.prediction.confidence,
            probabilities: response.data.prediction.probabilities,
            processingTime: response.data.metadata?.processing_time_ms,
            filename: response.data.metadata?.filename,
            heatmap: response.data.prediction.heatmap,
            details: `X-ray image classified as ${response.data.prediction.class} with ${(response.data.prediction.confidence * 100).toFixed(1)}% confidence.`
          };

          sessionStorage.setItem("analysisResult", JSON.stringify(result));
          setStatus("success");
        } else {
          throw new Error(response.data.message || "Prediction failed");
        }
      } catch (err: any) {
        console.error("Backend prediction failed:", err);
        const errorMsg = err.response?.data?.message ||
          "Failed to analyze image. Make sure the backend server is running.";
        setError(errorMsg);
        setStatus("error");
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "An unexpected error occurred.");
      setStatus("error");
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-6 relative overflow-hidden bg-black selection:bg-white selection:text-black">

      {/* Centered Area */}
      <div className="w-full max-w-4xl relative z-20 flex flex-col items-center justify-center flex-1">

        {/* Header Section */}
        <div className={`transition-all duration-700 ease-in-out w-full flex justify-center
          ${status === 'success' ? 'opacity-50 scale-90 mb-6' : 'mb-12'}`}>
          <Header />
        </div>

        {/* Interaction Area */}
        <div className="w-full relative flex flex-col items-center justify-center">

          {/* Upload Zone */}
          <div className={`transition-all duration-700 ease-in-out w-full flex justify-center
              ${status === "success" ? "opacity-0 pointer-events-none absolute scale-95" : "opacity-100 scale-100 relative z-10"}`}>
            <UploadZone
              onFileSelect={handleFileSelect}
              status={status === "success" ? "success" : status}
              error={error}
            />
          </div>
        </div>
      </div>
    </main>
  );
}
