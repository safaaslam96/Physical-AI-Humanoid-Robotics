import React, { useEffect, useState } from 'react';

const MyHighlights = () => {
  const [showSidebar, setShowSidebar] = useState(false);
  const [highlights, setHighlights] = useState<any[]>([]);

  // Load all highlights from localStorage
  const loadAllHighlights = () => {
    if (typeof window === 'undefined') return [];

    const allHighlights = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith('book_highlights_')) {
        try {
          const pageHighlights = JSON.parse(localStorage.getItem(key) || '[]');
          allHighlights.push(...pageHighlights);
        } catch (e) {
          console.error('Error loading highlights from key:', key, e);
        }
      }
    }
    return allHighlights;
  };

  // Remove all highlights for current page
  const removeAllHighlights = () => {
    if (typeof window === 'undefined') return;

    const pagePath = window.location.pathname;
    const key = `book_highlights_${pagePath}`;
    localStorage.removeItem(key);

    // Remove all highlight spans from the DOM
    const highlightElements = document.querySelectorAll('.book-highlight');
    highlightElements.forEach(el => {
      const parent = el.parentElement;
      if (parent) {
        parent.replaceChild(document.createTextNode(el.textContent || ''), el);
      }
    });

    // Refresh the highlights list
    setHighlights(loadAllHighlights());
    setShowSidebar(false);
  };

  // Remove a single highlight
  const removeHighlight = (id: string) => {
    const element = document.querySelector(`[data-highlight-id="${id}"]`) as HTMLElement;
    if (element) {
      const parent = element.parentElement;
      if (parent) {
        // Get only the direct text content, excluding any child elements like tooltips
        let originalText = '';
        for (let i = 0; i < element.childNodes.length; i++) {
          const node = element.childNodes[i];
          if (node.nodeType === Node.TEXT_NODE) {
            originalText += node.textContent;
          }
        }
        parent.replaceChild(document.createTextNode(originalText), element);
      }
    }

    // Update localStorage
    const pagePath = window.location.pathname;
    const key = `book_highlights_${pagePath}`;
    const saved = localStorage.getItem(key);
    const highlights = saved ? JSON.parse(saved) : [];
    const updatedHighlights = highlights.filter((h: any) => h.id !== id);
    localStorage.setItem(key, JSON.stringify(updatedHighlights));

    // Refresh the highlights list
    setHighlights(loadAllHighlights());
  };

  // Scroll to highlight
  const scrollToHighlight = (id: string) => {
    const element = document.querySelector(`[data-highlight-id="${id}"]`) as HTMLElement;
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      element.style.animation = 'highlight-pulse 1s ease-in-out';
      setTimeout(() => {
        element.style.animation = '';
      }, 1000);
    }
  };

  useEffect(() => {
    setHighlights(loadAllHighlights());

    // Listen for storage changes
    const handleStorageChange = () => {
      setHighlights(loadAllHighlights());
    };

    window.addEventListener('storage', handleStorageChange);

    // Log confirmation of highlighter button update
    console.log("Highlighter button updated - transparent, smaller, stacked above AI Chat, duplicates removed");

    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, []);

  return (
    <>
      {/* My Highlights floating button */}
      <button
        style={{
          position: 'fixed',
          bottom: '90px',
          right: '20px',
          zIndex: 9999,
          background: 'transparent !important',
          color: 'var(--ifm-color-primary, #8b5cf6)',
          border: 'none',
          borderRadius: '50%',
          width: '48px',
          height: '48px',
          fontSize: '16px',
          cursor: 'pointer',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
        onClick={() => setShowSidebar(true)}
        title="My Highlights"
      >
        📌
      </button>

      {/* My Highlights sidebar */}
      {showSidebar && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            zIndex: 9997,
          }}
          onClick={() => setShowSidebar(false)}
        >
          <div
            style={{
              position: 'fixed',
              top: 0,
              right: 0,
              height: '100vh',
              width: '350px',
              zIndex: 9998,
              backgroundColor: 'var(--ifm-background-surface-color, #ffffff)',
              boxShadow: '-4px 0 12px rgba(0, 0, 0, 0.15)',
              padding: '20px',
              overflowY: 'auto',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ margin: 0, color: 'var(--ifm-heading-color)' }}>My Highlights</h3>
              <button
                onClick={() => setShowSidebar(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '20px',
                  cursor: 'pointer',
                  color: 'var(--ifm-font-color-base)',
                }}
              >
                ✕
              </button>
            </div>

            <button
              onClick={removeAllHighlights}
              style={{
                background: '#e53e3e',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                padding: '8px 12px',
                marginBottom: '20px',
                cursor: 'pointer',
                fontSize: '14px',
                width: '100%',
              }}
            >
              Remove All Highlights on This Page
            </button>

            <div>
              {highlights.length === 0 ? (
                <p style={{ color: 'var(--ifm-font-color-base)' }}>No highlights yet. Select text and click the highlighter to start!</p>
              ) : (
                highlights.map((highlight, index) => (
                  <div
                    key={highlight.id || `highlight-${index}`}
                    style={{
                      padding: '10px',
                      marginBottom: '10px',
                      backgroundColor: document.documentElement.getAttribute('data-theme') === 'dark'
                        ? 'rgba(167, 139, 250, 0.4)'
                        : 'rgba(254, 240, 138, 0.6)',
                      borderRadius: '6px',
                      border: '1px solid var(--ifm-color-emphasis-200)',
                      cursor: 'pointer',
                      color: 'var(--ifm-font-color-base)',
                    }}
                    onClick={() => highlight.id && scrollToHighlight(highlight.id)}
                  >
                    <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
                      {highlight.text.substring(0, 50)}{highlight.text.length > 50 ? '...' : ''}
                    </div>
                    <div style={{ fontSize: '12px', color: 'var(--ifm-color-emphasis-600)' }}>
                      {new Date(highlight.timestamp).toLocaleDateString()}
                    </div>
                    <div style={{ fontSize: '12px', color: 'var(--ifm-color-emphasis-600)' }}>
                      {highlight.pagePath}
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        highlight.id && removeHighlight(highlight.id);
                      }}
                      style={{
                        background: '#e53e3e',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        padding: '4px 8px',
                        marginTop: '5px',
                        cursor: 'pointer',
                        fontSize: '12px',
                      }}
                    >
                      Remove
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        .book-highlight {
          transition: background-color 0.2s ease;
        }

        .book-highlight:hover {
          opacity: 0.9;
        }

        [data-theme='dark'] .book-highlight {
          background-color: rgba(167, 139, 250, 0.4) !important;
        }

        [data-theme='dark'] .text-highlighter-toolbar {
          background-color: var(--ifm-background-surface-color) !important;
          border-color: var(--ifm-color-primary) !important;
        }

        @keyframes highlight-pulse {
          0% { transform: scale(1); }
          50% { transform: scale(1.05); }
          100% { transform: scale(1); }
        }
      `}</style>
    </>
  );
};

export default MyHighlights;