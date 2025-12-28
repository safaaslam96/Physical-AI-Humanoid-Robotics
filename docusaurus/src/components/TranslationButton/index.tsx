import React, { useState, useEffect } from 'react';
import { translatePageContent, getCurrentPageContent, updatePageContent, restoreOriginalContent } from '../Translation/translationUtils';

const TranslationButton: React.FC = () => {
  const [isTranslated, setIsTranslated] = useState(false);
  const [originalContent, setOriginalContent] = useState('');

  const handleTranslate = async () => {
    try {
      if (!isTranslated) {
        // Translate to Urdu
        await translatePageContent('ur');
        setIsTranslated(true);
      } else {
        // Restore original content
        restoreOriginalContent();
        setIsTranslated(false);
      }
    } catch (error) {
      console.error('Translation error:', error);
      alert('Translation failed. Please try again.');
    }
  };

  return (
    <div style={{ marginBottom: '20px' }}>
      <button
        onClick={handleTranslate}
        style={{
          background: '#8b5cf6',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          padding: '10px 16px',
          fontSize: '14px',
          fontWeight: 'bold',
          cursor: 'pointer',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        }}
      >
        {isTranslated ? 'Switch to English' : 'Translate to Pure Urdu'}
      </button>
    </div>
  );
};

export default TranslationButton;