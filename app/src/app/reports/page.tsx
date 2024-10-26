'use client'

import Header from "../../components/Header";
import { ClipboardList, Activity, Calendar, Brain } from "lucide-react";

export default function Reports() {
  const upcomingFeatures = [
    {
      icon: <Activity className="w-8 h-8" />,
      title: "Mood Tracking",
      description: "Visual representations of your emotional patterns and trends over time."
    },
    {
      icon: <Brain className="w-8 h-8" />,
      title: "Mental Wellness Metrics",
      description: "Comprehensive analysis of your mental well-being indicators."
    },
    {
      icon: <Calendar className="w-8 h-8" />,
      title: "Progress Timeline",
      description: "Track your journey and milestone achievements."
    },
    {
      icon: <ClipboardList className="w-8 h-8" />,
      title: "Custom Reports",
      description: "Generate personalized reports based on your specific needs."
    }
  ];

  return (
    <div className="bg-bg-dark text-text-light min-h-screen flex flex-col font-sans">
      {/* Header */}
      <Header />

      {/* Main Content */}
      <main className="flex-grow flex flex-col items-center justify-center px-4 py-8 sm:px-8 md:px-12 lg:px-20">
        {/* Coming Soon Section */}
        <div className="text-center mb-12 max-w-2xl">
          <h1 className="text-4xl font-bold mb-6">Reports Dashboard Coming Soon</h1>
          <p className="text-text-muted text-lg mb-8">
            We're working on building comprehensive reporting tools to help you track and understand your mental well-being journey better.
          </p>
          <div className="inline-block relative">
            <div className="absolute inset-0 bg-primary-green opacity-20 blur-lg rounded-full"></div>
            <button className="relative bg-primary-green text-bg-dark py-3 px-8 rounded-full hover:bg-primary-blue transition-colors">
              Get Notified When Live
            </button>
          </div>
        </div>

        {/* Feature Preview Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 w-full max-w-6xl">
          {upcomingFeatures.map((feature, index) => (
            <div 
              key={index} 
              className="bg-bg-accent p-6 rounded-lg border border-gray-700 hover:border-primary-green transition-all duration-300 transform hover:scale-105 hover:shadow-lg"
            >
              <div className="text-primary-green mb-4 flex justify-center">
                {feature.icon}
              </div>
              <h3 className="text-lg font-semibold mb-2 text-center">{feature.title}</h3>
              <p className="text-text-muted text-center text-sm">
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        {/* Progress Indication */}
        <div className="mt-16 flex flex-col items-center">
          <div className="w-full max-w-md bg-bg-accent rounded-full h-2 mb-4">
            <div className="bg-primary-green h-2 rounded-full w-3/4"></div>
          </div>
          <p className="text-text-muted">
            Development Progress: <span className="text-primary-green">75%</span>
          </p>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-bg-accent py-6 text-center">
        <p>© 2024 hearU. All rights reserved.</p>
      </footer>
    </div>
  );
}