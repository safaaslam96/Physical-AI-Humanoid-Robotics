import React, { useState } from 'react';

const TranslateButton: React.FC = () => {
  const [isTranslating, setIsTranslating] = useState(false);
  const [currentLocale, setCurrentLocale] = useState('en');

  const handleTranslate = async () => {
    setIsTranslating(true);

    try {
      // Determine target locale
      const targetLocale = currentLocale === 'en' ? 'ur' : 'en';

      // Get current page content
      const pageContent = {
        title: document.title,
        content: document.body.innerText, // Simplified - in real app would get actual MDX content
        url: window.location.pathname,
      };

      // Call translation skill
      const response = await fetch('http://127.0.0.1:8001/api/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: JSON.stringify(pageContent),
          source_language: currentLocale,
          target_language: targetLocale,
          preserve_markdown: true,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        // In a real implementation, this would update the page with translated content
        // For now, we'll just switch the locale
        setCurrentLocale(targetLocale);

        // Update the UI direction for RTL languages like Urdu
        if (targetLocale === 'ur') {
          document.body.setAttribute('dir', 'rtl');
        } else {
          document.body.removeAttribute('dir');
        }
      }
    } catch (error) {
      console.error('Translation error:', error);
    } finally {
      setIsTranslating(false);
    }
  };

  return (
    <div style={{ marginBottom: '1rem' }}>
      <button
        onClick={handleTranslate}
        disabled={isTranslating}
        style={{
          backgroundColor: '#800000',
          color: 'white',
          border: 'none',
          padding: '8px 16px',
          borderRadius: '4px',
          cursor: isTranslating ? 'not-allowed' : 'pointer',
          opacity: isTranslating ? 0.7 : 1,
        }}
      >
        {isTranslating
          ? 'Translating...'
          : currentLocale === 'en'
            ? '.Translate to Urdu'
            : 'Translate to English'}
      </button>
    </div>
  );
};

export default TranslateButton;