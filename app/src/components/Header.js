'use client'
import React, { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';

const Header = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <header className="relative flex justify-between items-center py-6 px-8 bg-bg-accent">
      <Link href="/">
        <div className="flex items-center gap-4">
          <Image src="/headerLogo.png" width={72} height={55} alt="hearU Logo" />
        </div>
      </Link>

      <nav className="hidden md:flex gap-12">
        <Link href="/" className="hover:text-primary-green">
          Home
        </Link>
        <Link href="/about" className="hover:text-primary-green">
          About
        </Link>
        <Link href="/reports" className="hover:text-primary-green">
          Reports
        </Link>
        <Link href="/support" className="hover:text-primary-green">
          Support
        </Link>
      </nav>

      <Link href="/signin">
        <button className="bg-primary-green text-bg-dark py-2 px-5 rounded-md hover:bg-primary-blue hidden md:block">
          Sign In
        </button>
      </Link>

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
  );
};

export default Header;