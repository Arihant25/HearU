import Image from "next/image";
import Header from "../components/Header";

export default function Home() {
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
            <button className="mt-6 bg-primary-green text-bg-dark py-3 px-8 rounded-full hover:bg-primary-blue gradient-border">
              Get Started
            </button>
          </div>
        </div>

        {/* Right Content*/}
        <div className="md:w-1/2 mt-10 md:mt-0 flex justify-center ml-4 sm:ml-8 md:ml-12 lg:ml-16 xl:ml-20">
          <Image 
            src="/logo.png" 
            width={1000} 
            height={400} 
            alt="Mental Well-being Illustration" 
            style={{ width: '29vw', height: '21vw' }} 
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
