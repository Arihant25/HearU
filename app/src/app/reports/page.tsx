"use client";

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import Chart from 'chart.js/auto';
import Image from 'next/image';
import Link from 'next/link';

// Types
interface AnalysisTrends {
  total_messages: number;
  average_sentiment: number;
  category_distribution: { [key: string]: number };
  time_span: {
    start: string;
    end: string;
  };
}

// Cookie utility function
const getCookie = (name: string): string | null => {
  if (typeof document === 'undefined') return null;

  const cookies = document.cookie.split(';');
  for (let cookie of cookies) {
    const [cookieName, cookieValue] = cookie.split('=').map(c => c.trim());
    if (cookieName === name) {
      return cookieValue;
    }
  }
  return null;
};

const ReportsPage = () => {
  const sentimentChartRef = useRef<Chart | null>(null);
  const emotionsChartRef = useRef<Chart | null>(null);
  const router = useRouter();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [trends, setTrends] = useState<AnalysisTrends | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const userNumber = getCookie('hearU_user_number');
    const hasStarted = getCookie('hearU_started');

    if (!userNumber || !hasStarted) {
      router.push('/');
      return;
    }

    fetchAnalysisTrends(userNumber);
  }, [router]);

  const fetchAnalysisTrends = async (userId: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`http://localhost:8000/analyze_trends/${userId}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch analysis data');
      }

      const data = await response.json();
      setTrends(data);
      initializeCharts(data);
    } catch (error) {
      console.error('Error fetching analysis trends:', error);
      setError('Failed to load analysis data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const initializeCharts = (data: AnalysisTrends) => {
    if (sentimentChartRef.current) {
      sentimentChartRef.current.destroy();
    }
    if (emotionsChartRef.current) {
      emotionsChartRef.current.destroy();
    }

    const timeSpan = {
      start: new Date(data.time_span.start),
      end: new Date(data.time_span.end),
    };

    const dateLabels = generateDateLabels(timeSpan.start, timeSpan.end);

    // Sentiment Analysis Chart
    const sentimentCtx = document.getElementById('sentimentChart') as HTMLCanvasElement;
    if (sentimentCtx) {
      sentimentChartRef.current = new Chart(sentimentCtx, {
        type: 'line',
        data: {
          labels: dateLabels,
          datasets: [{
            label: 'Sentiment Score',
            data: Array(dateLabels.length).fill(data.average_sentiment),
            borderColor: '#FF5733',
            backgroundColor: 'rgba(255, 87, 51, 0.1)',
            tension: 0.4,
            fill: true
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              grid: {
                color: 'rgba(255, 255, 255, 0.1)'
              },
              ticks: {
                color: '#B3B3B3'
              }
            },
            x: {
              grid: {
                color: 'rgba(255, 255, 255, 0.1)'
              },
              ticks: {
                color: '#B3B3B3'
              }
            }
          }
        }
      }) as Chart;
    }

    // Categories/Emotions Chart
    const emotionsCtx = document.getElementById('emotionsChart') as HTMLCanvasElement;
    if (emotionsCtx) {
      const categories = Object.keys(data.category_distribution);
      const values = Object.values(data.category_distribution);

      emotionsChartRef.current = new Chart(emotionsCtx, {
        type: 'doughnut',
        data: {
          labels: categories,
          datasets: [{
            data: values,
            backgroundColor: [
              '#4CAF50', // anxiety
              '#FF9800', // depression
              '#2196F3', // academic_stress
              '#9C27B0', // relationship_issues
              '#F44336'  // career_confusion
            ]
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'right',
              labels: {
                color: '#B3B3B3'
              }
            }
          }
        }
      }) as Chart;
    }
  };

  const generateDateLabels = (start: Date, end: Date): string[] => {
    const labels = [];
    const current = new Date(start);
    while (current <= end) {
      labels.push(current.toLocaleDateString());
      current.setDate(current.getDate() + 1);
    }
    return labels;
  };

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <div className="bg-bg-dark text-text-light min-h-screen flex flex-col">
      <header className="relative flex justify-between items-center py-6 px-8 bg-bg-accent">
        <Link href="/">
          <div className="flex items-center gap-4">
            <Image src="/headerLogo.png" width={72} height={55} alt="hearU Logo" />
          </div>
        </Link>

        <nav className="hidden md:flex gap-12">
          <Link href="/chat" className="hover:text-primary-green">
            Chat
          </Link>
          <Link href="/support" className="hover:text-primary-green">
            Support
          </Link>
        </nav>

        <button
          className="md:hidden z-50"
          onClick={toggleMobileMenu}
          aria-label="Toggle mobile menu"
        >
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d={isMobileMenuOpen ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16m-7 6h7"}
            />
          </svg>
        </button>

        <div
          className={`fixed top-0 right-0 h-full w-64 bg-bg-accent shadow-lg transform transition-transform duration-300 ease-in-out md:hidden ${
            isMobileMenuOpen ? 'translate-x-0' : 'translate-x-full'
          }`}
        >
          <div className="flex flex-col items-center pt-24 h-full">
            <Link
              href="/chat"
              className="w-full py-4 text-center hover:text-primary-green hover:bg-black/5"
              onClick={toggleMobileMenu}
            >
              Chat
            </Link>
            <Link
              href="/support"
              className="w-full py-4 text-center hover:text-primary-green hover:bg-black/5"
              onClick={toggleMobileMenu}
            >
              Support
            </Link>
          </div>
        </div>
      </header>

      <div className="flex-grow p-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold mb-8">Your Mental Health Report</h2>

          {loading ? (
            <div className="flex justify-center items-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-green"></div>
            </div>
          ) : error ? (
            <div className="text-red-500 text-center p-4 bg-bg-accent rounded-lg">
              {error}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Sentiment Analysis Chart */}
              <div className="bg-bg-accent rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4">Sentiment Timeline</h3>
                <div className="h-64">
                  <canvas id="sentimentChart"></canvas>
                </div>
              </div>

              {/* Categories Distribution Chart */}
              <div className="bg-bg-accent rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4">Category Distribution</h3>
                <div className="h-64">
                  <canvas id="emotionsChart"></canvas>
                </div>
              </div>

              {/* Analysis Summary */}
              <div className="bg-bg-accent rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4">Analysis Summary</h3>
                <div className="space-y-4">
                  <p>Total Messages: {trends?.total_messages || 0}</p>
                  <p>Average Sentiment: {(trends?.average_sentiment || 0).toFixed(2)}</p>
                  <p>Time Period: {trends?.time_span.start.split('T')[0]} to {trends?.time_span.end.split('T')[0]}</p>
                </div>
              </div>

              {/* Recommendations */}
              <div className="bg-bg-accent rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-4">Recommendations</h3>
                <ul className="list-disc list-inside space-y-2">
                  {trends?.average_sentiment < 0 ? (
                    <>
                      <li>Consider speaking with a mental health professional</li>
                      <li>Practice daily mindfulness exercises</li>
                      <li>Maintain social connections</li>
                      <li>Establish a regular sleep schedule</li>
                      <li>Engage in physical activity</li>
                    </>
                  ) : (
                    <>
                      <li>Continue your positive practices</li>
                      <li>Share your progress with others</li>
                      <li>Set new personal growth goals</li>
                      <li>Maintain your support network</li>
                      <li>Celebrate your achievements</li>
                    </>
                  )}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ReportsPage;