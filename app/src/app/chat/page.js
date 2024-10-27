"use client";

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Header from '../../components/Header';
import Image from 'next/image';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const router = useRouter();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const fetchChatHistory = async (userId) => {
    try {
      const response = await fetch(`${API_URL}/history/${userId}`);
      if (!response.ok) throw new Error('Failed to fetch chat history');
      const data = await response.json();

      const formattedMessages = data.messages.map(msg => ({
        content: msg.message,
        isUser: true,
        timestamp: new Date(msg.timestamp)
      }));

      setMessages(formattedMessages);
    } catch (err) {
      setError('Failed to load chat history');
      console.error(err);
    }
  };

  const sendMessageToBackend = async (userId, message) => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_URL}/chat/${userId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: message }),
      });

      if (!response.ok) throw new Error('Failed to send message');

      const data = await response.json();
      return data.model_response;
    } catch (err) {
      setError('Failed to send message');
      console.error(err);
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const userNumber = getCookie('hearU_user_number');
    const hasStarted = getCookie('hearU_started');

    if (!userNumber || !hasStarted) {
      router.push('/');
      return;
    }

    fetchChatHistory(userNumber);
  }, [router]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    return () => {
      const userId = getCookie('hearU_user_number');
      if (userId) {
        fetch(`${API_URL}/close_conversation/${userId}`, {
          method: 'POST',
        }).catch(console.error);
      }
    };
  }, []);

  const addMessage = (content, isUser = true) => {
    setMessages(prev => [...prev, { content, isUser, timestamp: new Date() }]);
  };

  const handleSend = async () => {
    if (inputMessage.trim()) {
      const userId = getCookie('hearU_user_number');
      if (!userId) {
        router.push('/');
        return;
      }

      addMessage(inputMessage.trim(), true);
      setInputMessage('');

      const botResponse = await sendMessageToBackend(userId, inputMessage.trim());
      if (botResponse) {
        addMessage(botResponse, false);
      }
    } else if (!isRecording) {
      setIsRecording(true);
    } else {
      setIsRecording(false);
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

        {isMobileMenuOpen && (
          <div
            className="fixed inset-0 bg-black/20 md:hidden"
            onClick={toggleMobileMenu}
            aria-hidden="true"
          />
        )}
      </header>

      <div className="flex-grow flex flex-col max-w-4xl mx-auto w-full p-4">
        {error && (
          <div className="bg-red-500 text-white p-2 rounded mb-4">
            {error}
          </div>
        )}

        <div className="flex-grow bg-bg-accent rounded-lg p-4 mb-4 overflow-y-auto">
          {isLoading && (
            <div className="text-center p-2">
              <span className="animate-spin mr-2">⌛</span>
              Processing...
            </div>
          )}

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