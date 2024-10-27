"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Chart from "chart.js/auto";
import Image from "next/image";
import Link from "next/link";

// Define interfaces for the API response data
interface AnalysisData {
  total_messages: number;
  average_sentiment: number;
  category_distribution: { [key: string]: number };
  average_intensity: number;
  time_span: {
    start: string;
    end: string;
  };
  sentiment_progression: { [key: string]: number };
  intensity_progression: { [key: string]: number };
  top_keywords: [string, number][];
}

interface SuggestionResponse {
  suggestions: string;
}

const ReportsPage = () => {
  const sentimentChartRef = useRef<Chart | null>(null);
  const categoryChartRef = useRef<Chart | null>(null);
  const intensityChartRef = useRef<Chart | null>(null);
  const keywordChartRef = useRef<Chart | null>(null);
  const router = useRouter();

  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const userId = getCookie("hearU_user_number");
    if (!userId) {
      router.push("/");
      return;
    }

    fetchData(userId);
  }, [router]);

  const fetchData = async (userId: string) => {
    try {
      setLoading(true);

      // Fetch analysis data
      const analysisResponse = await fetch(
        `http://localhost:8000/analyze_trends/${userId}`
      );
      if (!analysisResponse.ok)
        throw new Error("Failed to fetch analysis data");
      const analysisData = await analysisResponse.json();
      setAnalysisData(analysisData);

      // Fetch suggestions
      const suggestionsResponse = await fetch(
        `http://localhost:8000/get_suggestions/${userId}`
      );
      if (!suggestionsResponse.ok)
        throw new Error("Failed to fetch suggestions");
      const suggestionsData: SuggestionResponse =
        await suggestionsResponse.json();
      setSuggestions(
        suggestionsData.suggestions.split("\n").filter((s) => s.trim())
      );

      initializeCharts(analysisData);
    } catch (error) {
      console.error("Error fetching data:", error);
      setError("Failed to load report data");
    } finally {
      setLoading(false);
    }
  };

  const initializeCharts = (data: AnalysisData) => {
    // Destroy existing charts
    [
      sentimentChartRef,
      categoryChartRef,
      intensityChartRef,
      keywordChartRef,
    ].forEach((ref) => {
      if (ref.current) ref.current.destroy();
    });

    // Sentiment Timeline Chart
    const sentimentCtx = document.getElementById(
      "sentimentChart"
    ) as HTMLCanvasElement;
    if (sentimentCtx) {
      const labels = Object.keys(data.sentiment_progression);
      const values = Object.values(data.sentiment_progression);

      sentimentChartRef.current = new Chart(sentimentCtx, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Sentiment Score",
              data: values,
              borderColor: "#4CAF50",
              backgroundColor: "rgba(76, 175, 80, 0.1)",
              tension: 0.4,
              fill: true,
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            title: { display: true, text: "Sentiment Over Time" },
          },
        },
      });
    }

    // Category Distribution Chart
    const categoryCtx = document.getElementById(
      "categoryChart"
    ) as HTMLCanvasElement;
    if (categoryCtx) {
      const categories = Object.keys(data.category_distribution);
      const values = Object.values(data.category_distribution);

      categoryChartRef.current = new Chart(categoryCtx, {
        type: "doughnut",
        data: {
          labels: categories,
          datasets: [
            {
              data: values,
              backgroundColor: [
                "#FF6384",
                "#36A2EB",
                "#FFCE56",
                "#4BC0C0",
                "#9966FF",
              ],
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            title: { display: true, text: "Category Distribution" },
          },
        },
      });
    }

    // Intensity Timeline Chart
    const intensityCtx = document.getElementById(
      "intensityChart"
    ) as HTMLCanvasElement;
    if (intensityCtx) {
      const labels = Object.keys(data.intensity_progression);
      const values = Object.values(data.intensity_progression);

      intensityChartRef.current = new Chart(intensityCtx, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Intensity Score",
              data: values,
              borderColor: "#FF9800",
              backgroundColor: "rgba(255, 152, 0, 0.1)",
              tension: 0.4,
              fill: true,
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            title: { display: true, text: "Intensity Over Time" },
          },
        },
      });
    }

    // Keywords Chart
    const keywordCtx = document.getElementById(
      "keywordChart"
    ) as HTMLCanvasElement;
    if (keywordCtx) {
      const [keywords, counts] = data.top_keywords.reduce(
        ([k, c], [keyword, count]) => [
          [...k, keyword],
          [...c, count],
        ],
        [[] as string[], [] as number[]]
      );

      keywordChartRef.current = new Chart(keywordCtx, {
        type: "bar",
        data: {
          labels: keywords,
          datasets: [
            {
              label: "Frequency",
              data: counts,
              backgroundColor: "#2196F3",
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            title: { display: true, text: "Top Keywords" },
          },
        },
      });
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-green"></div>
      </div>
    );
  }

  if (error) {
    return <div className="text-red-500 text-center p-4">{error}</div>;
  }

  return (
    <div className="bg-bg-dark text-text-light min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">
          Mental Health Analysis Report
        </h1>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Charts */}
          <div className="bg-bg-accent p-6 rounded-lg">
            <canvas id="sentimentChart"></canvas>
          </div>
          <div className="bg-bg-accent p-6 rounded-lg">
            <canvas id="categoryChart"></canvas>
          </div>
          <div className="bg-bg-accent p-6 rounded-lg">
            <canvas id="intensityChart"></canvas>
          </div>
          <div className="bg-bg-accent p-6 rounded-lg">
            <canvas id="keywordChart"></canvas>
          </div>

          {/* Summary Statistics */}
          <div className="bg-bg-accent p-6 rounded-lg">
            <h2 className="text-xl font-semibold mb-4">Summary</h2>
            <div className="space-y-2">
              <p>Total Messages: {analysisData?.total_messages}</p>
              <p>
                Average Sentiment: {analysisData?.average_sentiment.toFixed(2)}
              </p>
              <p>
                Average Intensity: {analysisData?.average_intensity.toFixed(2)}
              </p>
              <p>
                Time Period:{" "}
                {new Date(
                  analysisData?.time_span.start || ""
                ).toLocaleDateString()}{" "}
                -
                {new Date(
                  analysisData?.time_span.end || ""
                ).toLocaleDateString()}
              </p>
            </div>
          </div>

          {/* Suggestions */}
          <div className="bg-bg-accent p-6 rounded-lg">
            <h2 className="text-xl font-semibold mb-4">Recommendations</h2>
            <ul className="list-disc list-inside space-y-2">
              {suggestions.map((suggestion, index) => (
                <li key={index}>{suggestion}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

// Utility function to get cookie value
const getCookie = (name: string): string | null => {
  if (typeof document === "undefined") return null;
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop()?.split(";").shift() || null;
  return null;
};

export default ReportsPage;
