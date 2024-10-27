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

// Generate random number between min and max (inclusive)
const generateRandomNumber = (min = 1000, max = 9999) => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

export default function Home() {
  const router = useRouter();
  const [isReady, setIsReady] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const [userNumber, setUserNumber] = useState(null);

  // Check if user has already started and router is ready
  useEffect(() => {
    const savedNumber = getCookie('hearU_user_number');
    if (savedNumber) {
      setUserNumber(parseInt(savedNumber));
      setHasStarted(true);
    }
    setIsReady(true);
  }, []);

  const handleGetStarted = async () => {
    if (!isReady) return;

    // Generate random number if not already set
    if (!userNumber) {
      const randomNum = generateRandomNumber();
      setUserNumber(randomNum);
      setCookie('hearU_user_number', randomNum.toString(), 30); // Store for 30 days
    }

    // Set started cookie
    setCookie('hearU_started', 'true', 30);
    setHasStarted(true);

    try {
      await router.push('/chat');
    } catch (error) {
      console.error('Navigation error:', error);
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
          <div className="flex flex-col items-center space-y-4">
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
            {userNumber && (
              <p className="text-sm text-text-muted">
                Your unique number: {userNumber}
              </p>
            )}
          </div>
        </div>

<<<<<<< HEAD
        {/* Right Content*/}
        <div className="md:w-1/2 mt-10 md:mt-0 flex justify-center ml-4 sm:ml-8 md:ml-12 lg:ml-16 xl:ml-20">
          <Image 
            src="/logo.png" 
            width={1000} 
            height={400} 
            alt="Mental Well-being Illustration" 
            style={{ width: '29vw', height: '21vw' }} 
=======
        {/* Right Content */}
        <div className="md:w-1/2 mt-10 md:mt-0 flex justify-center">
          <Image
            src="/logo.png"
            width={1000}
            height={400}
            alt="Mental Well-being Illustration"
            style={{ width: '29vw', height: '21vw' }}
            priority
>>>>>>> 65c3201d36ab866f1c13f74d29e130f8f47dd92a
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