import React, { useState, useEffect } from 'react';

const HighlightsSidebar: React.FC = () => {
  const [showSidebar, setShowSidebar] = useState(false);
  const [highlights, setHighlights] = useState<any[]>([]);
  const [aiChatOpen, setAiChatOpen] = useState(false);

  // Check if dark mode is active
  const isDarkMode = () => {
    return document.documentElement.getAttribute('data-theme') === 'dark';
  };

  // Get all highlights across pages
  const getAllHighlights = (): any[] => {
    const allHighlights: any[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith('highlights_')) {
        try {
          const pageHighlights = JSON.parse(localStorage.getItem(key) || '[]');
          allHighlights.push(...pageHighlights.map((h: any) => ({
            ...h,
            pagePath: key.replace('highlights_', '')
          })));
        } catch (e) {
          console.error('Error loading highlights from key:', key, e);
        }
      }
    }
    return allHighlights;
  };

  // Load all highlights
  useEffect(() => {
    setHighlights(getAllHighlights());
  }, [showSidebar]);

  // Refresh highlights when sidebar is opened
  useEffect(() => {
    if (showSidebar) {
      setHighlights(getAllHighlights());
    }
  }, [showSidebar]);

  // Listen for AI Chat state changes
  useEffect(() => {
    const handleAIChatStateChange = (e: any) => {
      const newChatState = e.detail.isOpen;
      setAiChatOpen(newChatState);

      // Log visibility state changes
      if (newChatState) {
        console.log("Highlighter hidden on chat open");
      } else {
        console.log("Highlighter shown on chat close");
      }
    };

    document.addEventListener('aiChatStateChanged', handleAIChatStateChange);

    return () => {
      document.removeEventListener('aiChatStateChanged', handleAIChatStateChange);
    };
  }, []);

  return (
    <>
      {/* My Highlights Floating Button */}
      <button
        onClick={() => setShowSidebar(!showSidebar)}
        className="my-highlights-btn"
        style={{
          position: 'fixed',
          bottom: '100px',
          right: '20px',
          zIndex: 9999,
          background: 'transparent',
          color: 'var(--ifm-color-primary, #8b5cf6)',
          border: '1px solid var(--ifm-color-primary, #8b5cf6)',
          borderRadius: '50%',
          width: '48px',
          height: '48px',
          fontSize: '16px',
          cursor: 'pointer',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
          display: aiChatOpen ? 'none !important' : 'block !important',
        }}
        title="My Highlights"
      >
        📌
      </button>

      {/* Highlights Sidebar */}
      {showSidebar && (
        <div
          className="highlights-sidebar"
          style={{
            position: 'fixed',
            top: '0',
            right: '0',
            height: '100vh',
            width: '350px',
            zIndex: 9998,
            backgroundColor: 'var(--ifm-background-surface-color, #ffffff)',
            borderLeft: '1px solid var(--ifm-color-emphasis-300, #e2e8f0)',
            padding: '20px',
            overflowY: 'auto',
            boxShadow: '-4px 0 12px rgba(0, 0, 0, 0.15)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h3>My Highlights</h3>
            <button
              onClick={() => setShowSidebar(false)}
              style={{
                background: 'none',
                border: 'none',
                fontSize: '20px',
                cursor: 'pointer',
              }}
            >
              ✕
            </button>
          </div>
          <div>
            {highlights.length === 0 ? (
              <p>No highlights yet. Select text and click the highlighter to start!</p>
            ) : (
              highlights.map((highlight, index) => (
                <div
                  key={`${highlight.pagePath}-${highlight.id || `index-${index}`}`}
                  className="highlight-item"
                  style={{
                    padding: '10px',
                    marginBottom: '10px',
                    backgroundColor: isDarkMode()
                      ? '#a78bfa66' // Default dark mode color
                      : '#fef08a', // Default light mode color
                    borderRadius: '6px',
                    border: '1px solid var(--ifm-color-emphasis-300, #e2e8f0)',
                  }}
                >
                  <div
                    style={{
                      fontWeight: 'bold',
                      marginBottom: '5px',
                      cursor: 'pointer',
                    }}
                    onClick={() => {
                      // Navigate to the page where the highlight was made
                      window.location.href = highlight.pagePath;
                      setShowSidebar(false);
                    }}
                  >
                    {highlight.text.substring(0, 50)}{highlight.text.length > 50 ? '...' : ''}
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--ifm-color-emphasis-700)', marginBottom: '5px' }}>
                    {new Date(highlight.timestamp).toLocaleString()} • {highlight.pagePath}
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--ifm-color-emphasis-600)' }}>
                    {highlight.color} highlight
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </>
  );
};

export default HighlightsSidebar;