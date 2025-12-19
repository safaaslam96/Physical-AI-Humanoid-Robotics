import React, { useState, useEffect, useRef } from 'react';

interface HighlighterProps {
  children: React.ReactNode;
}

const Highlighter: React.FC<HighlighterProps> = ({ children }) => {
  const [selection, setSelection] = useState<Range | null>(null);
  const [showHighlighter, setShowHighlighter] = useState(false);
  const [highlightColor, setHighlightColor] = useState('#FFEB3B');
  const [highlights, setHighlights] = useState<Array<{id: string, text: string, color: string}>>([]);
  const [userId, setUserId] = useState<string | null>(null);
  const elementRef = useRef<HTMLDivElement>(null);

  const highlightColors = [
    { name: 'yellow', value: '#FFEB3B' },
    { name: 'green', value: '#4CAF50' },
    { name: 'blue', value: '#2196F3' },
    { name: 'orange', value: '#FF9800' },
    { name: 'pink', value: '#E91E63' },
  ];

  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('sessionToken');
    if (token) {
      setUserId('user_' + Math.random().toString(36).substr(2, 9)); // In real app, this would be the actual user ID
    }

    // Load saved highlights
    const savedHighlights = localStorage.getItem('highlights');
    if (savedHighlights) {
      setHighlights(JSON.parse(savedHighlights));
    }

    // Set up event listeners
    const handleSelection = () => {
      const sel = window.getSelection();
      if (sel && sel.toString().trim() !== '' && elementRef.current?.contains(sel.anchorNode)) {
        setSelection(sel.getRangeAt(0));
        setShowHighlighter(true);
      } else {
        setShowHighlighter(false);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('keyup', handleSelection);

    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('keyup', handleSelection);
    };
  }, []);

  useEffect(() => {
    if (selection && showHighlighter) {
      const rect = selection.getBoundingClientRect();
      const highlighter = document.getElementById('highlighter-popup');
      if (highlighter) {
        highlighter.style.top = `${rect.top + window.scrollY - 40}px`;
        highlighter.style.left = `${rect.left + window.scrollX + rect.width / 2 - 75}px`;
      }
    }
  }, [selection, showHighlighter]);

  const handleHighlight = () => {
    if (selection && userId) {
      const selectedText = selection.toString();
      const newHighlight = {
        id: `highlight_${Date.now()}`,
        text: selectedText,
        color: highlightColor,
      };

      const newHighlights = [...highlights, newHighlight];
      setHighlights(newHighlights);
      localStorage.setItem('highlights', JSON.stringify(newHighlights));

      // Send to backend
      fetch('http://127.0.0.1:8001/api/select-text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: selectedText,
          user_id: userId,
          chapter_id: window.location.pathname,
          highlight_color: highlightColor,
          context: selectedText,
        }),
      });

      setShowHighlighter(false);
      setSelection(null);
    }
  };

  return (
    <div ref={elementRef}>
      <div style={{ position: 'relative' }}>
        {children}
      </div>

      {showHighlighter && (
        <div
          id="highlighter-popup"
          style={{
            position: 'absolute',
            zIndex: 1000,
            backgroundColor: '#fff',
            border: '1px solid #ddd',
            borderRadius: '4px',
            boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
            padding: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
          }}
        >
          {highlightColors.map((color) => (
            <button
              key={color.name}
              onClick={() => setHighlightColor(color.value)}
              style={{
                width: '20px',
                height: '20px',
                backgroundColor: color.value,
                border: highlightColor === color.value ? '2px solid #000' : '1px solid #ccc',
                borderRadius: '50%',
                cursor: 'pointer',
              }}
              title={`Highlight with ${color.name}`}
            />
          ))}
          <button
            onClick={handleHighlight}
            style={{
              padding: '4px 8px',
              backgroundColor: '#800000',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Highlight
          </button>
        </div>
      )}

      {/* Render existing highlights */}
      {highlights.map((highlight) => (
        <div
          key={highlight.id}
          style={{
            backgroundColor: highlight.color,
            padding: '2px 4px',
            margin: '4px 0',
            borderRadius: '3px',
            border: '1px solid #ccc',
          }}
        >
          <strong>Highlighted:</strong> {highlight.text}
        </div>
      ))}
    </div>
  );
};

export default Highlighter;