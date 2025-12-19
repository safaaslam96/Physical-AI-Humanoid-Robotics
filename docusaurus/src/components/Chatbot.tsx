import React, { useState, useEffect, useRef } from 'react';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

const Chatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m your AI assistant for the Physical AI & Humanoid Robotics book. How can I help you today?',
      sender: 'bot',
      timestamp: new Date(),
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: input,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Get user ID if logged in
      const userId = localStorage.getItem('sessionToken') ? 'user_' + Math.random().toString(36).substr(2, 9) : null;

      // Call RAG API
      const response = await fetch('http://127.0.0.1:8001/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          user_id: userId,
          chapter_context: window.location.pathname,
          use_highlights_only: false, // User can toggle this
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const botMessage: Message = {
          id: Date.now().toString(),
          text: data.response,
          sender: 'bot',
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        const errorMessage: Message = {
          id: Date.now().toString(),
          text: 'Sorry, I encountered an error. Please try again.',
          sender: 'bot',
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: Message = {
        id: Date.now().toString(),
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
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
      sendMessage();
    }
  };

  return (
    <>
      {/* Floating Chat Button */}
      {!isOpen && (
        <button
          onClick={toggleChat}
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            backgroundColor: '#800000',
            color: 'white',
            border: 'none',
            fontSize: '24px',
            cursor: 'pointer',
            zIndex: 1000,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          }}
          aria-label="Open chat"
        >
          💬
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: '350px',
            height: '500px',
            backgroundColor: 'white',
            border: '1px solid #ddd',
            borderRadius: '8px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            zIndex: 1000,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {/* Header */}
          <div
            style={{
              backgroundColor: '#800000',
              color: 'white',
              padding: '12px',
              borderRadius: '8px 8px 0 0',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <h3 style={{ margin: 0 }}>AI Assistant</h3>
            <button
              onClick={toggleChat}
              style={{
                background: 'none',
                border: 'none',
                color: 'white',
                fontSize: '18px',
                cursor: 'pointer',
              }}
            >
              ×
            </button>
          </div>

          {/* Messages */}
          <div
            style={{
              flex: 1,
              padding: '12px',
              overflowY: 'auto',
              backgroundColor: '#f9f9f9',
            }}
          >
            {messages.map((message) => (
              <div
                key={message.id}
                style={{
                  marginBottom: '10px',
                  textAlign: message.sender === 'user' ? 'right' : 'left',
                }}
              >
                <div
                  style={{
                    display: 'inline-block',
                    padding: '8px 12px',
                    borderRadius: '18px',
                    backgroundColor: message.sender === 'user' ? '#800000' : '#e9ecef',
                    color: message.sender === 'user' ? 'white' : '#333',
                  }}
                >
                  {message.text}
                </div>
                <div
                  style={{
                    fontSize: '0.7em',
                    color: '#666',
                    marginTop: '4px',
                  }}
                >
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            ))}
            {isLoading && (
              <div style={{ textAlign: 'left', marginBottom: '10px' }}>
                <div
                  style={{
                    display: 'inline-block',
                    padding: '8px 12px',
                    borderRadius: '18px',
                    backgroundColor: '#e9ecef',
                    color: '#333',
                  }}
                >
                  Thinking...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div style={{ padding: '12px', borderTop: '1px solid #ddd' }}>
            <div style={{ display: 'flex' }}>
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about the book..."
                style={{
                  flex: 1,
                  padding: '8px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  marginRight: '8px',
                }}
                disabled={isLoading}
              />
              <button
                onClick={sendMessage}
                disabled={isLoading || !input.trim()}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#800000',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: isLoading || !input.trim() ? 'not-allowed' : 'pointer',
                }}
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Chatbot;