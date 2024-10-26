'use client'

import Image from "next/image";
import Link from "next/link";
import Header from "../../components/Header";
import { Heart, Shield, Users, Target, MessageCircle, Lightbulb } from "lucide-react";

export default function About() {
  const missionPoints = [
    {
      icon: <Heart className="w-8 h-8" />,
      title: "Compassionate Support",
      description: "We believe in providing empathetic and understanding support for everyone's mental health journey."
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Safe Space",
      description: "Creating a secure and judgment-free environment where you can express yourself freely."
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: "Community Focused",
      description: "Building a supportive community where everyone feels heard and valued."
    },
    {
      icon: <Target className="w-8 h-8" />,
      title: "Personalized Approach",
      description: "Tailoring our support to meet your individual needs and goals."
    }
  ];

  const features = [
    {
      icon: <MessageCircle className="w-8 h-8" />,
      title: "24/7 Support",
      description: "Access to support whenever you need it, day or night."
    },
    {
      icon: <Lightbulb className="w-8 h-8" />,
      title: "Innovative Tools",
      description: "Modern solutions for mental wellness tracking and improvement."
    }
  ];

  return (
    <div className="bg-bg-dark text-text-light min-h-screen flex flex-col font-sans">
      <Header />

      <main className="flex-grow px-4 py-8 sm:px-8 md:px-12 lg:px-20">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold mb-6">About hearU</h1>
          <p className="text-text-muted text-lg max-w-2xl mx-auto">
            We're dedicated to making mental health support accessible, 
            personal, and effective for everyone who needs it.
          </p>
        </div>

        {/* Mission Section */}
        <div className="mb-20">
          <h2 className="text-2xl font-bold text-center mb-12">Our Mission</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {missionPoints.map((point, index) => (
              <div 
                key={index}
                className="bg-bg-accent p-6 rounded-lg border border-gray-700 hover:border-primary-green transition-all duration-300 transform hover:scale-105"
              >
                <div className="text-primary-green mb-4 flex justify-center">
                  {point.icon}
                </div>
                <h3 className="text-lg font-semibold mb-2 text-center">
                  {point.title}
                </h3>
                <p className="text-text-muted text-center text-sm">
                  {point.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Features Section */}
        <div className="mb-20">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-2xl font-bold text-center mb-12">What We Offer</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {features.map((feature, index) => (
                <div 
                  key={index}
                  className="bg-bg-accent p-8 rounded-lg border border-gray-700 hover:border-primary-green transition-all duration-300 transform hover:scale-105"
                >
                  <div className="text-primary-green mb-4 flex justify-center">
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-semibold mb-3 text-center">
                    {feature.title}
                  </h3>
                  <p className="text-text-muted text-center">
                    {feature.description}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Join Us Section */}
        <div className="text-center mb-16">
          <div className="max-w-2xl mx-auto">
            <h2 className="text-2xl font-bold mb-6">Join Our Community</h2>
            <p className="text-text-muted mb-8">
              Be part of a supportive community that understands and values mental health. 
              Together, we can create a more understanding and compassionate world.
            </p>
            <div className="inline-block relative">
              <div className="absolute inset-0 bg-primary-green opacity-20 blur-lg rounded-full"></div>
              <button className="relative bg-primary-green text-bg-dark py-3 px-8 rounded-full hover:bg-primary-blue transition-colors">
                Get Started Today
              </button>
            </div>
          </div>
        </div>
      </main>

      <footer className="bg-bg-accent py-6 text-center">
        <p>© 2024 hearU. All rights reserved.</p>
      </footer>
    </div>
  );
}