"use client"
import Image from "next/image";
import Header from "../components/Header";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

// Cookie utility functions
const setCookie = (name, value, days = 7) => {
  const expires = new Date();
  expires.setDate(expires.getDate() + days);
  document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/`;
};

const getCookie = (name) => {
  const cookies = document.cookie.split(';');
  for (let cookie of cookies) {
    const [cookieName, cookieValue] = cookie.split('=').map(c => c.trim());
    if (cookieName === name) {
      return cookieValue;
    }
  }
  return null;
};

export default function Home() {
  const router = useRouter();
  const [isReady, setIsReady] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);

  // Check if user has already started and router is ready
  useEffect(() => {
    const userStarted = getCookie('hearU_started');
    if (userStarted) {
      setHasStarted(true);
    }
    setIsReady(true);
  }, []);

  const handleGetStarted = async () => {
    if (!isReady) return;

    // Set cookie to remember user has started
    setCookie('hearU_started', 'true', 30); // Cookie expires in 30 days
    setHasStarted(true);
    
    try {
      await router.push('/chat');
    } catch (error) {
      console.error('Navigation error:', error);
      // Optionally handle navigation error here
    }
  };

  return (
    <div className="bg-bg-dark text-text-light min-h-screen flex flex-col font-sans">
      {/* Header */}
      <Header />
      
      {/* Main Section */}
      <main className="flex-grow flex flex-col md:flex-row items-center px-4 py-4 sm:px-8 sm:py-8 md:px-12 md:py-16 lg:px-20 lg:py-20">
        {/* Left Content */}
        <div className="md:w-1/2 space-y-8">
          <h2 className="text-5xl font-extrabold leading-snug">
            Empower Your Mental Well-being
          </h2>
          <p className="text-xl text-text-muted text-justify" style={{ width: '100%' }}>
            At hearU, we offer a secure space for self-reflection, support, and mental well-being insights. Join us in taking a step towards understanding and managing your mental health with confidence.
          </p>
          <div className="flex justify-center">
            <button 
              onClick={handleGetStarted}
              disabled={!isReady}
              className={`mt-6 py-3 px-8 rounded-full gradient-border transition-all duration-300 ${
                !isReady 
                  ? 'opacity-50 cursor-not-allowed'
                  : hasStarted 
                    ? 'bg-primary-blue text-bg-dark hover:bg-primary-green' 
                    : 'bg-primary-green text-bg-dark hover:bg-primary-blue'
              }`}
            >
              {hasStarted ? 'Continue Journey' : 'Get Started'}
            </button>
          </div>
        </div>
        
        {/* Right Content */}
        <div className="md:w-1/2 mt-10 md:mt-0 flex justify-center">
          <Image 
            src="/logo.png"
            width={1000}
            height={400}
            alt="Mental Well-being Illustration"
            style={{ width: '29vw', height: '21vw' }}
            priority
          />
        </div>
      </main>
      
      {/* Footer */}
      <footer className="bg-bg-accent py-6 text-center">
        <p>© 2024 hearU. All rights reserved.</p>
      </footer>
    </div>
  );
}