import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

const AIChatPopup: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm your AI assistant for Physical AI & Humanoid Robotics. How can I help you today?",
      sender: 'ai',
      timestamp: new Date(),
    },
    {
      id: '2',
      text: "Ask me about ROS2, Gazebo simulation, NVIDIA Isaac, or any chapter from the book!",
      sender: 'ai',
      timestamp: new Date(),
    },
    {
      id: '3',
      text: "I can explain complex concepts, provide code examples, or help with exercises.",
      sender: 'ai',
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const toggleChat = () => {
    const newOpenState = !isOpen;
    setIsOpen(newOpenState);

    // Dispatch custom event to notify other components about chat state change
    const event = new CustomEvent('aiChatStateChanged', {
      detail: { isOpen: newOpenState }
    });
    document.dispatchEvent(event);

    // Log visibility state changes
    if (newOpenState) {
      console.log("Highlighter hidden on chat open");
    } else {
      console.log("Highlighter shown on chat close");
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Check if this is a question about selected/highlighted text
    const isAskAboutSelectedText = inputValue.startsWith('Explain this concept: "');

    // Extract selected text if present in the query
    let selectedText = null;
    if (isAskAboutSelectedText) {
      // Extract the selected text from the input
      const match = inputValue.match(/"([^"]*)"/);
      if (match && match[1]) {
        selectedText = match[1];
      }
    }

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Get backend URL from environment or use default
      const backendUrl = typeof window !== 'undefined' && (window as any).BACKEND_URL
        ? (window as any).BACKEND_URL
        : process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

      // Call the RAG backend API at /api/chat
      const response = await fetch(`${backendUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputValue,
          selected_text: selectedText,
          from_selected_text: !!selectedText
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();

      const aiMessage: Message = {
        id: Date.now().toString(),
        text: data.response,
        sender: 'ai',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error calling RAG API:', error);

      // Fallback message if API call fails
      const errorMessage: Message = {
        id: Date.now().toString(),
        text: "Sorry, I'm having trouble connecting to the knowledge base. Please try again later.",
        sender: 'ai',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e as any);
    }
  };

  // Listen for custom event from text highlighter
  useEffect(() => {
    const handleAskAIEvent = (e: any) => {
      const selectedText = e.detail.text;
      if (selectedText) {
        // Open the chat if it's not already open
        if (!isOpen) {
          setIsOpen(true);
        }

        // Set the input value to a question about the selected text
        setInputValue(`Explain this concept: "${selectedText}"`);
      }
    };

    document.addEventListener('askAIAboutText', handleAskAIEvent);

    return () => {
      document.removeEventListener('askAIAboutText', handleAskAIEvent);
    };
  }, [isOpen]);

  return (
    <>
      {/* Floating Chat Button */}
      {!isOpen && (
        <motion.button
          onClick={toggleChat}
          className="ai-chat-float-button"
          aria-label="Open AI Chat"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            color: '#4c1d95',
            border: '1px solid #8b5cf6',
            fontSize: '16px',
            cursor: 'pointer',
            zIndex: 1000,
            boxShadow: '0 4px 20px rgba(139, 92, 246, 0.4), 0 0 20px rgba(139, 92, 246, 0.2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          🤖
        </motion.button>
      )}

      {/* Chat Popup */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="ai-chat-popup"
            initial={{ opacity: 0, y: 100, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 100, scale: 0.9 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300, duration: 0.3 }}
            style={{
              position: 'fixed',
              bottom: '20px',
              right: '20px',
              width: '400px',
              maxWidth: 'calc(100vw - 40px)',
              maxHeight: '70vh',
              zIndex: 1000,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            {/* Glassmorphism Container */}
            <div
              className="ai-chat-container"
              style={{
                width: '100%',
                height: '100%',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                background: 'rgba(30, 41, 59, 0.85)',
                border: '1px solid rgba(139, 92, 246, 0.3)',
                borderRadius: '20px',
                boxShadow: '0 8px 32px rgba(139, 92, 246, 0.3), 0 0 20px rgba(139, 92, 246, 0.2)',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              {/* Header */}
              <div
                className="ai-chat-header"
                style={{
                  padding: '16px 20px',
                  borderBottom: '1px solid rgba(139, 92, 246, 0.2)',
                  position: 'relative',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  background: 'rgba(15, 23, 42, 0.7)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ fontSize: '20px' }}>🤖</div>
                  <h3
                    style={{
                      margin: 0,
                      fontSize: '16px',
                      fontWeight: '600',
                      color: 'white',
                    }}
                  >
                    AI Assistant
                  </h3>
                </div>
                <button
                  onClick={toggleChat}
                  style={{
                    background: 'rgba(15, 23, 42, 0.7)',
                    border: '1px solid rgba(139, 92, 246, 0.3)',
                    borderRadius: '50%',
                    width: '32px',
                    height: '32px',
                    color: 'white',
                    fontSize: '18px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'background 0.2s',
                  }}
                  onMouseEnter={(e) => {
                    (e.target as HTMLElement).style.background = 'rgba(15, 23, 42, 0.9)';
                  }}
                  onMouseLeave={(e) => {
                    (e.target as HTMLElement).style.background = 'rgba(15, 23, 42, 0.7)';
                  }}
                >
                  ×
                </button>
              </div>

              {/* Messages */}
              <div
                className="ai-chat-messages"
                style={{
                  flex: 1,
                  padding: '16px',
                  overflowY: 'auto',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '12px',
                }}
              >
                {messages.map((message) => (
                  <motion.div
                    key={message.id || `message-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`}
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ duration: 0.3 }}
                    style={{
                      display: 'flex',
                      justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start',
                      width: '100%',
                    }}
                  >
                    <div
                      style={{
                        maxWidth: '85%',
                        padding: '12px 16px',
                        borderRadius: message.sender === 'user'
                          ? '18px 4px 18px 18px'
                          : '4px 18px 18px 18px',
                        backgroundColor: message.sender === 'user'
                          ? 'rgba(139, 92, 246, 0.8)'
                          : 'rgba(30, 41, 59, 0.8)',
                        color: message.sender === 'user' ? 'white' : 'white',
                        backdropFilter: 'blur(4px)',
                        WebkitBackdropFilter: 'blur(4px)',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
                        wordWrap: 'break-word',
                        wordBreak: 'break-word',
                      }}
                    >
                      {message.text}
                      <div
                        style={{
                          fontSize: '10px',
                          opacity: 0.7,
                          marginTop: '4px',
                          textAlign: 'right',
                        }}
                      >
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    </div>
                  </motion.div>
                ))}

                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ duration: 0.3 }}
                    style={{
                      display: 'flex',
                      justifyContent: 'flex-start',
                      width: '100%',
                    }}
                  >
                    <div
                      style={{
                        maxWidth: '85%',
                        padding: '12px 16px',
                        borderRadius: '4px 18px 18px 18px',
                        backgroundColor: '#e2e8f0',
                        color: '#1e293b',
                        backdropFilter: 'blur(4px)',
                        WebkitBackdropFilter: 'blur(4px)',
                        border: '1px solid rgba(139, 92, 246, 0.1)',
                        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
                      }}
                    >
                      <div className="typing-indicator" style={{ display: 'flex', gap: '4px' }}>
                        <span style={{
                          display: 'inline-block',
                          width: '8px',
                          height: '8px',
                          borderRadius: '50%',
                          backgroundColor: 'white',
                          opacity: 0.6,
                          animation: 'bounce 1.4s infinite both'
                        }}>•</span>
                        <span style={{
                          display: 'inline-block',
                          width: '8px',
                          height: '8px',
                          borderRadius: '50%',
                          backgroundColor: 'white',
                          opacity: 0.6,
                          animation: 'bounce 1.4s infinite both',
                          animationDelay: '0.2s'
                        }}>•</span>
                        <span style={{
                          display: 'inline-block',
                          width: '8px',
                          height: '8px',
                          borderRadius: '50%',
                          backgroundColor: 'white',
                          opacity: 0.6,
                          animation: 'bounce 1.4s infinite both',
                          animationDelay: '0.4s'
                        }}>•</span>
                      </div>
                    </div>
                  </motion.div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <form
                onSubmit={handleSendMessage}
                style={{
                  padding: '16px',
                  borderTop: '1px solid rgba(139, 92, 246, 0.2)',
                  backgroundColor: 'rgba(15, 23, 42, 0.5)',
                }}
              >
                <div style={{
                  display: 'flex',
                  gap: '8px',
                }}>
                  <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask about Physical AI..."
                    style={{
                      flex: 1,
                      padding: '12px 16px',
                      borderRadius: '24px',
                      border: '1px solid rgba(139, 92, 246, 0.3)',
                      backgroundColor: 'rgba(15, 23, 42, 0.7)',
                      color: 'white',
                      fontSize: '14px',
                      backdropFilter: 'blur(4px)',
                      WebkitBackdropFilter: 'blur(4px)',
                    }}
                    disabled={isLoading}
                  />
                  <motion.button
                    type="submit"
                    disabled={isLoading || !inputValue.trim()}
                    style={{
                      width: '44px',
                      height: '44px',
                      borderRadius: '50%',
                      backgroundColor: '#8b5cf6',
                      color: 'white',
                      border: 'none',
                      cursor: isLoading || !inputValue.trim() ? 'not-allowed' : 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '16px',
                    }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    disabled={isLoading || !inputValue.trim()}
                  >
                    ➤
                  </motion.button>
                </div>
              </form>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Dark Mode Styles */}
      <style dangerouslySetInnerHTML={{__html: `
        [data-theme='dark'] .ai-chat-container {
          background: rgba(15, 23, 42, 0.7) !important;
          border: 1px solid rgba(139, 92, 246, 0.3) !important;
          box-shadow: 0 8px 32px rgba(139, 92, 246, 0.3) !important;
        }

        [data-theme='dark'] .ai-chat-header {
          border-bottom: 1px solid rgba(139, 92, 246, 0.2) !important;
        }

        [data-theme='dark'] .ai-chat-messages {
          color: white !important;
        }

        [data-theme='dark'] .ai-chat-messages [data-sender="ai"] div {
          background: rgba(30, 41, 59, 0.8) !important;
          color: white !important;
        }

        [data-theme='dark'] input {
          color: white !important;
        }

        input::placeholder {
          color: rgba(30, 41, 59, 0.6) !important;
        }

        [data-theme='dark'] input::placeholder {
          color: rgba(255, 255, 255, 0.6) !important;
        }

        @keyframes bounce {
          0%, 80%, 100% {
            transform: scale(0);
          }
          40% {
            transform: scale(1);
          }
        }

        /* Responsive styles */
        @media (max-width: 480px) {
          .ai-chat-popup {
            width: calc(100vw - 20px) !important;
            bottom: 10px !important;
            right: 10px !important;
          }
        }
      `}} />
    </>
  );
};

export default AIChatPopup;