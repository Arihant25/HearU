import Image from "next/image";
import Link from "next/link";

export default function Login() {
  return (
    <div className="h-screen flex items-center justify-center bg-background font-sans">
      <div className="w-full max-w-md p-8 bg-gray-800 rounded-lg shadow-lg space-y-12 relative">
        {/* Back Button */}
        <Link href="/signin" className="absolute top-4 left-4 text-primary-green hover:text-button-hover transition duration-300">
          &larr; Back
        </Link>

        <div className="flex flex-col items-center mb-6 space-y-8">
          <Image
            src="/logo.png"
            width={150}
            height={150}
            alt="HearU Logo"
            className="mb-4"
          />
          <div className="flex flex-col items-center mb-6 space-y-4">
            <h1 className="text-3xl font-bold text-primary-green">Hello There!</h1>
            <p className="text-center text-text-muted">Your journey to mental well-being begins here!</p>
          </div>
        </div>

        <form className="space-y-6">
          <div>
            <label htmlFor="username" className="block text-text-light mb-2">Username</label>
            <input
              type="text"
              id="username"
              className="w-full p-3 bg-gray-700 text-text-light border border-gray-600 rounded focus:outline-none focus:ring-2 focus:ring-primary-blue"
              placeholder="Enter your username"
              required
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-text-light mb-2">Password</label>
            <input
              type="password"
              id="password"
              className="w-full p-3 bg-gray-700 text-text-light border border-gray-600 rounded focus:outline-none focus:ring-2 focus:ring-primary-blue"
              placeholder="Enter your password"
              required
            />
          </div>

          <div className="flex justify-center items-center">
            <button type="submit" className="signinButton">
              Sign Up
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}