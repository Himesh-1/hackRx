import React, { useState, useEffect, useRef } from 'react';
import { Send, Moon, Sun, FileText, CheckCircle, XCircle, DollarSign, BookOpen, Sparkles, Bot, User, Clock, Shield, Copy, Trash2, Check } from 'lucide-react';

const DocQueryAI = () => {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Load from localStorage
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : window.matchMedia('(prefers-color-scheme: dark)').matches;
  });
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [responses, setResponses] = useState([]);
  const [typingText, setTypingText] = useState('');
  const [copied, setCopied] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const exampleQueries = [
    {
      text: "46M, knee surgery, Pune, 3-month-old policy",
      category: "Surgical Claim",
      icon: Shield
    },
    {
      text: "32F, maternity claim, Chennai, premium policy",
      category: "Maternity",
      icon: CheckCircle
    },
    {
      text: "55M, cardiac procedure, Mumbai, basic coverage",
      category: "Cardiac Care",
      icon: FileText
    },
    {
      text: "28F, dental treatment, Bangalore, 1-year-old policy",
      category: "Dental",
      icon: BookOpen
    }
  ];

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [responses, isLoading]);

  // Save dark mode preference
  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(isDarkMode));
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  const handleSubmit = async () => {
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    const currentQuery = query;
    setQuery('');

    try {
      const response = await fetch('http://localhost:8001/process-query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: currentQuery }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      // Defensive: handle both snake_case and camelCase, and fallback for missing fields
      const newResponse = {
        id: Date.now(),
        query: currentQuery,
        decision: result.decision || result.Decision || 'Unknown',
        amount: result.amount || result.payout_amount || 0,
        justification: result.justification || result.reason || '',
        clauses: result.clauses || result.referenced_clauses || [],
        confidence: result.confidence || result.confidence_score || 0.8,
        processingTime: result.processing_time || result.processingTime || null,
      };

      setResponses(prev => [newResponse, ...prev]);
    } catch (error) {
      console.error("Error processing query:", error);
      // Optionally, display an error message to the user
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (example) => {
    setQuery(example.text);
    inputRef.current?.focus();
  };

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const clearChat = () => {
    setResponses([]);
    setQuery('');
  };

  return (
    <div className={`min-h-screen transition-all duration-500 ${
      isDarkMode 
        ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white' 
        : 'bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 text-gray-900'
    }`}>
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className={`absolute -top-40 -right-40 w-96 h-96 rounded-full opacity-10 blur-3xl ${
          isDarkMode ? 'bg-blue-600' : 'bg-blue-400'
        } animate-pulse`}></div>
        <div className={`absolute -bottom-40 -left-40 w-96 h-96 rounded-full opacity-10 blur-3xl ${
          isDarkMode ? 'bg-purple-600' : 'bg-purple-400'
        } animate-pulse`} style={{ animationDelay: '2s' }}></div>
      </div>

      {/* Header */}
      <header className={`sticky top-0 z-50 backdrop-blur-xl border-b transition-all duration-300 ${
        isDarkMode 
          ? 'bg-gray-900/95 border-gray-700/50 shadow-lg shadow-gray-900/20' 
          : 'bg-white/95 border-gray-200/50 shadow-lg shadow-blue-900/10'
      }`}>
        <div className="w-full px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className={`p-3 rounded-2xl shadow-lg ${
                isDarkMode 
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 shadow-blue-900/30' 
                  : 'bg-gradient-to-r from-blue-500 to-purple-600 shadow-blue-500/30'
              }`}>
                <FileText className="h-7 w-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl sm:text-3xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
                  DocQuery AI
                </h1>
                <p className={`text-xs sm:text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  Intelligent Document Analysis
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              {responses.length > 0 && (
                <button
                  onClick={clearChat}
                  className={`p-2 rounded-lg text-sm transition-all hover:scale-105 ${
                    isDarkMode ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-600'
                  }`}
                  aria-label="Clear chat"
                >
                  <Trash2 className="h-5 w-5" />
                </button>
              )}
              <button
                onClick={toggleTheme}
                className={`p-3 rounded-xl transition-all hover:scale-105 active:scale-95 ${
                  isDarkMode 
                    ? 'hover:bg-gray-700 text-gray-300' 
                    : 'hover:bg-gray-100 text-gray-600'
                }`}
                aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
              >
                {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative w-full px-4 sm:px-6 lg:px-8 pt-8 pb-32 max-w-6xl mx-auto">
        {/* Welcome Screen */}
        {responses.length === 0 && !isLoading && (
          <div className="text-center py-16">
            <div className={`inline-flex p-4 rounded-3xl mb-6 shadow-2xl ${
              isDarkMode 
                ? 'bg-gray-800/50 border border-gray-700/50 backdrop-blur-xl' 
                : 'bg-white/80 border border-white/50 backdrop-blur-xl'
            }`}>
              <Sparkles className={`h-12 w-12 ${isDarkMode ? 'text-blue-400' : 'text-blue-500'}`} />
            </div>
            <h2 className={`text-3xl sm:text-4xl lg:text-5xl font-bold mb-4 ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>
              Transform Documents into{' '}
              <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
                Insights
              </span>
            </h2>
            <p className={`text-lg mb-10 max-w-2xl mx-auto ${
              isDarkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              Ask intelligent questions about policy coverage, claims processing, and eligibility criteria using natural language.
            </p>

            {/* Example Queries */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 w-full max-w-5xl mx-auto">
              {exampleQueries.map((example, index) => {
                const Icon = example.icon;
                return (
                  <button
                    key={index}
                    onClick={() => handleExampleClick(example)}
                    className={`group p-5 rounded-2xl text-left transition-all duration-300 hover:scale-105 active:scale-95 focus:outline-none focus:ring-2 focus:ring-blue-500/50 ${
                      isDarkMode 
                        ? 'bg-gray-800/50 border border-gray-700/50 backdrop-blur-xl hover:bg-gray-700/50' 
                        : 'bg-white/80 border border-white/50 shadow-lg hover:shadow-xl backdrop-blur-xl'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <Icon className={`h-6 w-6 ${isDarkMode ? 'text-blue-400' : 'text-blue-500'}`} />
                      <span className={`text-xs px-2.5 py-1 rounded-full ${
                        isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-600'
                      }`}>
                        {example.category}
                      </span>
                    </div>
                    <p className="text-sm font-medium leading-tight">{example.text}</p>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className={`rounded-3xl p-7 mb-6 backdrop-blur-xl animate-fade-in-down ${
            isDarkMode 
              ? 'bg-gray-800/50 border border-gray-700/50 shadow-xl' 
              : 'bg-white/80 border border-white/50 shadow-2xl'
          }`}>
            <div className="flex items-center space-x-4 mb-5">
              <div className="relative">
                <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
                <Bot className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 h-5 w-5 text-white" />
              </div>
              <div>
                <p className="font-medium">Analyzing your query...</p>
                <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Extracting policy insights
                </p>
              </div>
            </div>
            {typingText && (
              <div className={`p-4 rounded-xl text-sm leading-relaxed ${
                isDarkMode ? 'bg-gray-700/40' : 'bg-gray-50/60'
              }`}>
                {typingText}
                <span className="animate-pulse ml-1">|</span>
              </div>
            )}
          </div>
        )}

        {/* Responses */}
        <div className="space-y-8">
          {responses.map((response) => (
            <div
              key={response.id}
              className={`rounded-3xl p-6 sm:p-8 transition-all duration-500 opacity-100 transform animate-fade-in-up ${
                isDarkMode 
                  ? 'bg-gray-800/50 border border-gray-700/50 shadow-2xl' 
                  : 'bg-white/90 border border-white/50 shadow-2xl'
              }`}
            
            >
              {/* User Query */}
              <div className={`mb-6 p-4 sm:p-5 rounded-2xl ${
                isDarkMode ? 'bg-gray-700/50' : 'bg-gray-50/70'
              }`}>
                <div className="flex items-center space-x-2 mb-2">
                  <User className="h-5 w-5 text-blue-500" />
                  <span className="text-sm font-medium opacity-80">Your Query</span>
                </div>
                <p className="font-medium text-base sm:text-lg">{response.query}</p>
              </div>

              {/* AI Response */}
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6 gap-4">
                <div className="flex items-center space-x-3">
                  <Bot className="h-6 w-6 text-purple-500" />
                  <span className="font-semibold text-lg">AI Analysis</span>
                </div>
                <div className="flex items-center space-x-3 text-sm">
                  <div className="flex items-center space-x-1 text-gray-500">
                    <Clock className="h-4 w-4" />
                    <span>{response.processingTime?.toFixed(1)}s</span>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                    response.confidence > 0.9 
                      ? 'bg-green-100 text-green-700' 
                      : response.confidence > 0.75 
                      ? 'bg-yellow-100 text-yellow-700' 
                      : 'bg-red-100 text-red-700'
                  }`}>
                    {(response.confidence * 100).toFixed(0)}% Conf
                  </span>
                </div>
              </div>

              {/* Decision */}
              <div className="flex items-center space-x-4 mb-6">
                <div className={`p-3 rounded-2xl ${
                  response.decision === 'Approved' 
                    ? 'bg-green-100 text-green-600' 
                    : 'bg-red-100 text-red-600'
                }`}>
                  {response.decision === 'Approved' 
                    ? <CheckCircle className="h-7 w-7" /> 
                    : <XCircle className="h-7 w-7" />
                  }
                </div>
                <div>
                  <p className="text-sm opacity-75">Final Decision</p>
                  <p className={`font-bold text-xl ${
                    response.decision === 'Approved' ? 'text-green-500' : 'text-red-500'
                  }`}>
                    {response.decision}
                  </p>
                </div>
              </div>

              {/* Coverage Amount */}
              {response.amount > 0 && (
                <div className="flex items-center space-x-4 mb-6">
                  <div className="p-3 rounded-2xl bg-blue-100 text-blue-600">
                    <DollarSign className="h-7 w-7" />
                  </div>
                  <div>
                    <p className="text-sm opacity-75">Approved Coverage</p>
                    <p className="font-bold text-xl text-blue-500">
                      ₹{response.amount.toLocaleString('en-IN')}
                    </p>
                  </div>
                </div>
              )}

              {/* Justification */}
              <div className="mb-6">
                <h3 className="font-semibold text-lg mb-3 flex items-center space-x-2">
                  <BookOpen className="h-5 w-5 text-purple-500" />
                  <span>Justification</span>
                </h3>
                <div className={`p-4 sm:p-5 rounded-2xl leading-relaxed text-sm sm:text-base ${
                  isDarkMode ? 'bg-gray-700/30' : 'bg-gray-50/60'
                } relative`}>
                  <p>{response.justification}</p>
                  <button
                    onClick={() => handleCopy(response.justification)}
                    className={`absolute top-3 right-3 p-2 rounded-lg text-sm transition-all ${
                      isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
                    }`}
                    aria-label="Copy justification"
                  >
                    {copied ? <Check className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              {/* Clauses */}
              {response.clauses && response.clauses.length > 0 && (
                <div>
                  <h3 className="font-semibold text-lg mb-3 flex items-center space-x-2">
                    <FileText className="h-5 w-5 text-indigo-500" />
                    <span>Referenced Clauses</span>
                  </h3>
                  <div className="grid grid-cols-1 gap-3">
                    {response.clauses.map((clause, idx) => (
                      <div
                        key={idx}
                        className={`p-3 rounded-xl border text-sm ${
                          isDarkMode 
                            ? 'bg-gray-700/20 border-gray-600 hover:bg-gray-700/40' 
                            : 'bg-gray-50/60 border-gray-200 hover:bg-white'
                        } transition-colors`}
                      >
                        {typeof clause === 'string' ? clause : (clause.clause_text || clause.text || JSON.stringify(clause))}
                        {clause.source_document ? ` (Source: ${clause.source_document})` : ''}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Fixed Input Bar */}
      <div className={`fixed bottom-0 left-0 right-0 border-t backdrop-blur-xl z-40 transition-all ${
        isDarkMode 
          ? 'bg-gray-900/95 border-gray-700/50 shadow-2xl shadow-black/30' 
          : 'bg-white/95 border-gray-200/50 shadow-2xl shadow-blue-200/30'
      }`}>
        <div className="w-full px-4 sm:px-6 lg:px-8 py-4 max-w-6xl mx-auto">
          <div className="flex space-x-3">
            <textarea
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about coverage, claims, eligibility..."
              disabled={isLoading}
              rows="1"
              className={`flex-1 px-5 py-4 rounded-2xl border transition-all focus:outline-none focus:ring-4 text-base resize-none overflow-hidden ${
                isDarkMode 
                  ? 'bg-gray-800 border-gray-600 text-white placeholder-gray-400 focus:border-blue-500 focus:ring-blue-500/20' 
                  : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-blue-500 focus:ring-blue-500/20'
              } ${isLoading ? 'opacity-60 cursor-not-allowed' : 'hover:border-blue-400'}`}
              style={{ maxHeight: '120px' }}
              onInput={(e) => {
                e.target.style.height = 'auto';
                e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
              }}
            />
            <button
              onClick={handleSubmit}
              disabled={!query.trim() || isLoading}
              className={`px-6 py-4 rounded-2xl font-medium flex items-center space-x-2 transition-all ${
                query.trim() && !isLoading
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-lg hover:shadow-xl hover:scale-105 active:scale-95'
                  : isDarkMode 
                    ? 'bg-gray-700 text-gray-400' 
                    : 'bg-gray-200 text-gray-400'
              }`}
            >
              <Send className={`h-5 w-5 ${isLoading ? 'animate-spin' : ''}`} />
              <span className="hidden sm:inline text-sm">{isLoading ? 'Sending...' : 'Ask'}</span>
            </button>
          </div>
          <p className="text-xs text-center mt-2 opacity-60">
            Press <kbd className="px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded text-xs">Enter</kbd> to send, <kbd className="px-2 py-1 bg-gray-200 dark:bg-gray-700 rounded text-xs">Shift+Enter</kbd> for new line
          </p>
        </div>
      </div>

      {/* Optional: Add this to your index.css or use inline styles */}
      <style jsx>{`
        @keyframes fade-in-up {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fade-in-up {
          animation: fade-in-up 0.5s ease-out forwards;
        }
        .animate-fade-in-down {
          animation: fade-in-up 0.4s ease-out forwards;
          transform: rotateX(10deg);
        }
      `}</style>
    </div>
  );
};

export default DocQueryAI;