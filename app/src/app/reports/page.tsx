"use client";

import { useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Chart from 'chart.js/auto';

// Cookie utility function (reusing from your home page)
const getCookie = (name) => {
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
  const sentimentChartRef = useRef(null);
  const emotionsChartRef = useRef(null);
  const router = useRouter();

  useEffect(() => {
    // Check authentication
    const userNumber = getCookie('hearU_user_number');
    const hasStarted = getCookie('hearU_started');

    if (!userNumber || !hasStarted) {
      router.push('/');
      return;
    }

    // Initialize charts if authenticated
    initializeCharts();
  }, [router]);

  const initializeCharts = () => {
    // Destroy existing charts if they exist
    if (sentimentChartRef.current) {
      sentimentChartRef.current.destroy();
    }
    if (emotionsChartRef.current) {
      emotionsChartRef.current.destroy();
    }

    // Sentiment Analysis Chart
    const sentimentCtx = document.getElementById('sentimentChart');
    if (sentimentCtx) {
      sentimentChartRef.current = new Chart(sentimentCtx, {
        type: 'line',
        data: {
          labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
          datasets: [{
            label: 'Mood Score',
            data: [65, 59, 80, 81, 56, 75, 85],
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
      });
    }

    // Emotions Chart
    const emotionsCtx = document.getElementById('emotionsChart');
    if (emotionsCtx) {
      emotionsChartRef.current = new Chart(emotionsCtx, {
        type: 'doughnut',
        data: {
          labels: ['Joy', 'Anxiety', 'Neutral', 'Sadness', 'Anger'],
          datasets: [{
            data: [30, 20, 25, 15, 10],
            backgroundColor: [
              '#4CAF50',
              '#FF9800',
              '#2196F3',
              '#9C27B0',
              '#F44336'
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
      });
    }
  };

  return (
    <div className="bg-bg-dark text-text-light min-h-screen flex flex-col">
      {/* Navigation */}
      <nav className="bg-bg-accent p-4">
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold">hear you</h1>
          <div className="space-x-6">
            <a href="/chat" className="hover:text-primary-green">Chat</a>
            <a href="/report" className="hover:text-primary-green">Report</a>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="flex-grow p-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold mb-8">Your Mental Health Report</h2>

          {/* Charts Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Sentiment Analysis Chart */}
            <div className="bg-bg-accent rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">Weekly Mood Trends</h3>
              <div className="h-64">
                <canvas id="sentimentChart"></canvas>
              </div>
            </div>

            {/* Emotions Chart */}
            <div className="bg-bg-accent rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">Emotion Distribution</h3>
              <div className="h-64">
                <canvas id="emotionsChart"></canvas>
              </div>
            </div>

            {/* Additional Stats or Information */}
            <div className="bg-bg-accent rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">Weekly Summary</h3>
              <div className="space-y-4">
                <p>Average Mood Score: 71.5</p>
                <p>Most Frequent Emotion: Joy</p>
                <p>Improvement from Last Week: +5%</p>
              </div>
            </div>

            {/* Recommendations */}
            <div className="bg-bg-accent rounded-lg p-6">
              <h3 className="text-xl font-semibold mb-4">Recommendations</h3>
              <ul className="list-disc list-inside space-y-2">
                <li>Consider meditation to manage anxiety levels</li>
                <li>Schedule regular physical activity</li>
                <li>Maintain consistent sleep schedule</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReportsPage;