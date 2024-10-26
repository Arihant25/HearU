'use client'
import Image from 'next/image'
import Link from 'next/link'
import { signIn } from 'next-auth/react'

export default function SignIn() {
  const handleGoogleSignIn = () => {
    signIn('google', { callbackUrl: '/dashboard' })
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-bg-accent p-4">
      <div className="w-full max-w-md text-center">
        {/* Logo */}
        <Link href="/" className="inline-block mb-8">
          <Image
            src="/headerLogo.png"
            width={100}
            height={76}
            alt="hearU Logo"
            className="mx-auto"
          />
        </Link>

        {/* Sign In Card */}
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h1 className="text-2xl font-semibold mb-6">Welcome to hearU</h1>
          
          {/* Google Sign In Button */}
          <button
            onClick={handleGoogleSignIn}
            className="w-full flex items-center justify-center gap-3 bg-white border border-gray-300 rounded-md px-4 py-3 text-gray-700 font-medium hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-green transition-all"
          >
            <Image
              src="/google-icon.svg"
              width={20}
              height={20}
              alt="Google logo"
            />
            Sign in with Google
          </button>
        </div>
      </div>
    </div>
  )
}