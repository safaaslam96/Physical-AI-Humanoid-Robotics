import React, { useState, useEffect, useRef } from 'react';
import styles from './chatbot.module.css';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

const ChatbotPopup: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Toggle chatbot open/close
  const toggleChatbot = () => {
    setIsOpen(!isOpen);
  };

  // Close chatbot
  const closeChatbot = () => {
    setIsOpen(false);
  };

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Handle sending a message
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Call the backend API to get AI response
      const response = await fetch('/api/v1/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputValue,
          session_id: 'chatbot-session', // Use a session ID for continuity
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from AI');
      }

      const data = await response.json();

      // Add AI response
      const aiMessage: Message = {
        id: Date.now().toString(),
        content: data.response,
        sender: 'ai',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Chat error:', error);

      // Add error message
      const errorMessage: Message = {
        id: Date.now().toString(),
        content: 'Sorry, I encountered an error. Please try again.',
        sender: 'ai',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Initialize with a welcome message
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      const welcomeMessage: Message = {
        id: 'welcome',
        content: 'Hello! I\'m your AI assistant for Physical AI and Humanoid Robotics. How can I help you today?',
        sender: 'ai',
        timestamp: new Date(),
      };
      setMessages([welcomeMessage]);
    }
  }, [isOpen]);

  return (
    <>
      {/* Chatbot toggle button - this will be styled via navbar config */}
      <button
        id="chatbot-toggle"
        className={`${styles.chatbotToggle} ${isOpen ? styles.hidden : ''}`}
        onClick={toggleChatbot}
        aria-label="Open AI Chatbot"
      >
        🤖 AI Chat
      </button>

      {/* Chatbot popup */}
      {isOpen && (
        <div className={styles.chatbotPopup}>
          <div className={styles.chatbotHeader}>
            <div className={styles.chatbotTitle}>
              <span className={styles.chatbotIcon}>🤖</span>
              <span>AI Assistant</span>
            </div>
            <button
              className={styles.closeButton}
              onClick={closeChatbot}
              aria-label="Close chatbot"
            >
              ×
            </button>
          </div>

          <div className={styles.chatbotBody}>
            <div className={styles.messagesContainer}>
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`${styles.message} ${styles[message.sender]}`}
                >
                  <div className={styles.messageContent}>
                    {message.content}
                  </div>
                  <div className={styles.messageTime}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className={`${styles.message} ${styles.ai}`}>
                  <div className={styles.messageContent}>
                    <div className={styles.typingIndicator}>
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          <form onSubmit={handleSendMessage} className={styles.chatbotInputForm}>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type your message..."
              className={styles.chatbotInput}
              disabled={isLoading}
              aria-label="Type your message to the AI assistant"
            />
            <button
              type="submit"
              className={styles.sendButton}
              disabled={isLoading || !inputValue.trim()}
              aria-label="Send message"
            >
              ➤
            </button>
          </form>
        </div>
      )}
    </>
  );
};

export default ChatbotPopup;