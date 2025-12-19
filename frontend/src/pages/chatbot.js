import React from 'react';
import Layout from '@theme/Layout';
import { useState, useEffect } from 'react';

export default function ChatbotPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Mock chat functionality for demonstration
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { id: Date.now(), text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Simulate API call to backend
    setTimeout(() => {
      const botMessage = {
        id: Date.now() + 1,
        text: `I received your message: "${input}". This is a demo response from the RAG chatbot. In the full implementation, this would connect to your backend API at http://localhost:8000/api/v1/chat/ to provide answers based on the book content.`,
        sender: 'bot'
      };
      setMessages(prev => [...prev, botMessage]);
      setIsLoading(false);
    }, 1000);
  };

  return (
    <Layout title="AI Chatbot" description="Interactive RAG Chatbot for Physical AI and Humanoid Robotics Book">
      <div style={{
        maxWidth: '800px',
        margin: '0 auto',
        padding: '2rem',
        minHeight: '80vh',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <h1>AI-Powered Chatbot</h1>
        <p>Ask questions about Physical AI and Humanoid Robotics. The chatbot will answer based on the book content.</p>

        <div style={{
          flex: 1,
          border: '1px solid #ccc',
          borderRadius: '8px',
          padding: '1rem',
          marginBottom: '1rem',
          maxHeight: '400px',
          overflowY: 'auto',
          backgroundColor: '#f9f9f9'
        }}>
          {messages.length === 0 ? (
            <div style={{ fontStyle: 'italic', color: '#666' }}>
              Welcome! Ask me anything about Physical AI and Humanoid Robotics. I can answer based on the book content.
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                style={{
                  marginBottom: '1rem',
                  textAlign: message.sender === 'user' ? 'right' : 'left'
                }}
              >
                <div
                  style={{
                    display: 'inline-block',
                    padding: '0.5rem 1rem',
                    borderRadius: '12px',
                    backgroundColor: message.sender === 'user' ? '#007cba' : '#e9ecef',
                    color: message.sender === 'user' ? 'white' : 'black',
                    maxWidth: '80%'
                  }}
                >
                  {message.text}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div style={{ textAlign: 'left', marginBottom: '1rem' }}>
              <div style={{
                display: 'inline-block',
                padding: '0.5rem 1rem',
                borderRadius: '12px',
                backgroundColor: '#e9ecef',
                color: 'black'
              }}>
                Thinking...
              </div>
            </div>
          )}
        </div>

        <form onSubmit={handleSendMessage} style={{ display: 'flex', gap: '0.5rem' }}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the book content..."
            style={{
              flex: 1,
              padding: '0.5rem',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
            disabled={isLoading}
          />
          <button
            type="submit"
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#007cba',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: isLoading ? 'not-allowed' : 'pointer'
            }}
            disabled={isLoading || !input.trim()}
          >
            Send
          </button>
        </form>

        <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '4px' }}>
          <h3>How it works</h3>
          <p>
            This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions based on the book content.
            In the full implementation, it would connect to the backend API at <code>http://localhost:8000</code>
            to retrieve relevant information from the vector database and generate accurate responses.
          </p>
        </div>
      </div>
    </Layout>
  );
}