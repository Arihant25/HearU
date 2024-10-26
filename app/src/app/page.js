import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div className="h-screen flex flex-row font-sans">
      <div className="md:w-1/2 bg-custom-blue-500 hidden md:flex items-center justify-center ">
        <Image
          src="/logo.png"
          width={500}
          height={500}
          alt="Picture of the author"
        />
      </div>
      <div className="md:w-1/2 md:bg-white text-black p-4 xbg-custom-blue-500">
        <nav className=" border-violet-200 flex flex-row justify-between items-center">
          <div className="flex flex-row gap-2">
            <Link href="">home</Link>
            <Link href="">about</Link>
            <Link href="">services</Link>
            <Link href="">blogs</Link>
          </div>
          <button className="rounded md:bg-custom-blue-500 md:text-white bg-white p-2  ">
            Sign in
          </button>
        </nav>

       

        <div className=" h-5/6 flex flex-row items-center mt-5">


        <div className="flex flex-col items-center justify-center md:mt-24 md:w-1/2 mx-auto">
        




          <h1 className="text-3xl font-bold text-center">Welcome to hearU</h1>
          <h3 className="text-sm font-light text-center">Your journey to mental well-being begins here.</h3>
          <p className="text-center mt-10">At hearYou, we believe that everyone deserves a safe space to express themselves, be understood, and find comfort. Our platform is designed to provide you with the support and tools you need to navigate life's challenges and improve your mental health.</p>
          <Image
          src="/logo.png"
          width={200}
          height={200}
          alt="Picture of the author"
          className="m-auto mt-5 md:hidden"
        />
          <button className="rounded md:bg-custom-blue-500 p-2 w-full mt-10 md:text-white bg-white">Get Started</button>
        </div>
      </div>
    </div>
        </div>
  );
}
