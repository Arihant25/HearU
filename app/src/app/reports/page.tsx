"use client";

import { useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Chart from 'chart.js/auto';
import Image from 'next/image';
import Link from 'next/link';
import { useState } from 'react';

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
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

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

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

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
            <header className="relative flex justify-between items-center py-6 px-8 bg-bg-accent">
        <Link href="/">
          <div className="flex items-center gap-4">
            <Image src="/headerLogo.png" width={72} height={55} alt="hearU Logo" />
          </div>
        </Link>

        <nav className="hidden md:flex gap-12">
          <Link href="/reports" className="hover:text-primary-green">
            Reports
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
              d={isMobileMenuOpen
                ? "M6 18L18 6M6 6l12 12"
                : "M4 6h16M4 12h16m-7 6h7"
              }
            >
            </path>
          </svg>
        </button>

        <div
          className={`fixed top-0 right-0 h-full w-64 bg-bg-accent shadow-lg transform transition-transform duration-300 ease-in-out md:hidden ${
            isMobileMenuOpen ? 'translate-x-0' : 'translate-x-full'
          }`}
        >
          <div className="flex flex-col items-center pt-24 h-full">
            <Link
              href="/"
              className="w-full py-4 text-center hover:text-primary-green hover:bg-black/5"
              onClick={toggleMobileMenu}
            >
              Home
            </Link>
            <Link
              href="/about"
              className="w-full py-4 text-center hover:text-primary-green hover:bg-black/5"
              onClick={toggleMobileMenu}
            >
              About
            </Link>
            <Link
              href="/reports"
              className="w-full py-4 text-center hover:text-primary-green hover:bg-black/5"
              onClick={toggleMobileMenu}
            >
              Reports
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

        {isMobileMenuOpen && (
          <div
            className="fixed inset-0 bg-black/20 md:hidden"
            onClick={toggleMobileMenu}
            aria-hidden="true"
          />
        )}
      </header>


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