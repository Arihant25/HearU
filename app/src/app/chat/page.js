"use client";

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Header from '../../components/Header';
import Image from 'next/image';
import Link from 'next/link';

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

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const messagesEndRef = useRef(null);
  const router = useRouter();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  useEffect(() => {
    // Check authentication
    const userNumber = getCookie('hearU_user_number');
    const hasStarted = getCookie('hearU_started');

    if (!userNumber || !hasStarted) {
      router.push('/');
      return;
    }
  }, [router]);

  useEffect(() => {
    // Scroll to bottom whenever messages change
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addMessage = (content, isUser = true) => {
    setMessages(prev => [...prev, { content, isUser, timestamp: new Date() }]);
  };

  const handleSend = () => {
    if (inputMessage.trim()) {
      addMessage(inputMessage.trim(), true);
      setInputMessage('');

      // Simulate bot response
      setTimeout(() => {
        addMessage("This is a sample response from hearU.", false);
      }, 1000);
    } else if (!isRecording) {
      // Start recording
      setIsRecording(true);
      // Add your voice recording logic here
    } else {
      // Stop recording
      setIsRecording(false);
      // Add your stop recording logic here
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
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
          <Link href="/signin">
            <button className="bg-primary-green text-bg-dark py-2 px-5 rounded-md hover:bg-primary-blue mt-8">
              Sign In
            </button>
          </Link>
        </div>
      </div>

      {/* Overlay */}
      {isMobileMenuOpen && (
        <div
          className="fixed inset-0 bg-black/20 md:hidden"
          onClick={toggleMobileMenu}
          aria-hidden="true"
        />
      )}
    </header>

      {/* Chat Container */}
      <div className="flex-grow flex flex-col max-w-4xl mx-auto w-full p-4">
        {/* Messages Area */}
        <div className="flex-grow bg-bg-accent rounded-lg p-4 mb-4 overflow-y-auto">
          <div className="space-y-4">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[70%] rounded-lg p-3 ${
                    msg.isUser
                      ? 'bg-primary-green text-bg-dark'
                      : 'bg-primary-blue text-bg-dark'
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="flex gap-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            className="flex-grow p-3 rounded-lg bg-bg-accent text-text-light focus:outline-none focus:ring-2 focus:ring-primary-green"
          />
          <button
            onClick={handleSend}
            className="p-3 rounded-lg bg-primary-green text-bg-dark hover:bg-primary-blue transition-colors"
          >
            <span className="material-icons">
              {inputMessage.trim() ? 'send' : isRecording ? 'stop' : 'mic'}
            </span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;