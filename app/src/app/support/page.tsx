'use client'

import Header from "../../components/Header";
import { useState } from "react";
import { ChevronDown, MessageCircle, Phone, Mail, Book } from "lucide-react";

export default function Support() {
  const [openFaq, setOpenFaq] = useState<number | null>(null);

    const faqs = [
    {
      question: "How do I get started with hearU?",
      answer: "Getting started is with hearU is very easy! Simply click the 'Sign In' button in the top right corner to create an account. Once registered, you can begin exploring our LLaMa powered chatbot assistant and delve into the detailed reports it generates based on your interaction."
    },
    {
      question: "Is my data private and secure?",
      answer: "Yes, we take your privacy very seriously. The very first step in our data pipelining is to tokenize and encrypt the data."
    },
    {
      question: "Can I track my progress over time?",
      answer: "Yes! hearU provides detailed progress tracking through our reports system. You can view your emotional trends and wellness metrics."
    },
    {
      question: "What kind of support does hearU offer?",
      answer: "hearU offers 24/7 support through our chatbot assistant."
    },
    {
      question: "Are there any subscription plans?",
      answer: "No, we at hearU believe in universal access of free and world-class mental health help. So, all our services are free and just a few clicks away."
    },
    {
      question: "Can I use hearU on my mobile device?",
      answer: "Yes, hearU is fully responsive and can be accessed on any mobile device through your web browser."
    },
    {
      question: "How often are the reports updated?",
      answer: "Reports are updated in real-time based on your interactions with the chatbot. You can view the latest data anytime from your dashboard."
    }
  ];

  const supportResources = [
    {
      icon: <Phone className="w-6 h-6" />,
      title: "Suicide Hotline",
      description: "Monday to Saturday 10AM - 8PM",
      availability: "+91 91529-87821"
    },
    {
      icon: <Mail className="w-6 h-6" />,
      title: "Email Support",
      description: "Get detailed assistance via email",
      availability: "OTSM@live.com"
    },
    {
      icon: <Book className="w-6 h-6" />,
      title: "Resource Library",
      description: "Self-help guides and materials",
      availability: "www.mhanational.org/self-help-tools"
    }
  ];

  return (
    <div className="bg-bg-dark text-text-light min-h-screen flex flex-col font-sans">

      <Header />

      {/* Main Content */}
      <main className="flex-grow px-4 py-8 sm:px-8 md:px-12 lg:px-20">
        {/* Support Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">How Can We Help You?</h1>
          <p className="text-text-muted text-lg max-w-2xl mx-auto">
            We're here to support your mental well-being journey. Browse through our resources
            or reach out directly for assistance.
          </p>
        </div>

        {/* Support Resources Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-16 space-y-12">
          {/* Add empty divs to fill the remaining spaces */}
          {Array.from({ length: 1 }).map((_, index) => (
            <div key={`empty-${index}`} className="bg-transparent p-6"></div>
          ))}

          {supportResources.map((resource, index) => (
            <div key={index} className="bg-bg-accent p-6 rounded-xl hover:bg-opacity-80 transition-all border border-gray-100 flex flex-col items-center text-center">
              <div className="text-primary-green mb-4">{resource.icon}</div>
              <h3 className="text-lg font-semibold mb-2">{resource.title}</h3>
              <p className="text-text-muted mb-2">{resource.description}</p>
              <p className="text-primary-green text-sm font-bold">{resource.availability}</p>
            </div>
          ))}

          {/* Add empty divs to fill the remaining spaces */}
          {Array.from({ length: 1 }).map((_, index) => (
            <div key={`empty-${index}`} className="bg-transparent p-6"></div>
          ))}
        </div>

        {/* FAQ Section */}
        <div className="max-w-3xl mx-auto mb-16 mt-8">
          <h2 className="text-2xl font-bold mb-8 text-center">Frequently Asked Questions</h2>
          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <div key={index} className="bg-bg-accent rounded-lg">
                <button
                  className="w-full px-6 py-4 text-left flex justify-between items-center"
                  onClick={() => setOpenFaq(openFaq === index ? null : index)}
                >
                  <span className="font-medium">{faq.question}</span>
                  <ChevronDown
                    className={`w-5 h-5 transition-transform ${
                      openFaq === index ? "transform rotate-180" : ""
                    }`}
                  />
                </button>
                {openFaq === index && (
                  <div className="px-6 py-4 border-t border-gray-700">
                    <p className="text-text-muted">{faq.answer}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-bg-accent py-6 text-center">
        <p>© 2024 hearU. All rights reserved.</p>
      </footer>
    </div>
  );
}