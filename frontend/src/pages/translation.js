import React from 'react';
import Layout from '@theme/Layout';
import { useState } from 'react';

export default function TranslationPage() {
  const [originalText, setOriginalText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [language, setLanguage] = useState('ur');

  const handleTranslate = async (e) => {
    e.preventDefault();
    if (!originalText.trim() || isTranslating) return;

    setIsTranslating(true);

    // Simulate API call to translation service
    setTimeout(() => {
      // This is a mock translation - in reality, this would call the backend
      setTranslatedText(`[This is a simulated translation to ${language.toUpperCase()}]. ${originalText} - This is a demonstration of the translation feature. In the full implementation, this would connect to the backend API at http://localhost:8000/api/v1/translation to provide professional Urdu translation while preserving technical accuracy.`);
      setIsTranslating(false);
    }, 1500);
  };

  return (
    <Layout title="Content Translation" description="Translate book content to Urdu for enhanced accessibility">
      <div style={{
        maxWidth: '800px',
        margin: '0 auto',
        padding: '2rem'
      }}>
        <h1>Content Translation</h1>
        <p>Translate book content to Urdu for enhanced accessibility and understanding.</p>

        <form onSubmit={handleTranslate} style={{ marginTop: '1rem' }}>
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
              Select Language
            </label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              style={{
                padding: '0.5rem',
                border: '1px solid #ccc',
                borderRadius: '4px'
              }}
            >
              <option value="ur">Urdu</option>
            </select>
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
              Original Text
            </label>
            <textarea
              value={originalText}
              onChange={(e) => setOriginalText(e.target.value)}
              placeholder="Enter text to translate..."
              style={{
                width: '100%',
                padding: '0.5rem',
                border: '1px solid #ccc',
                borderRadius: '4px',
                minHeight: '120px'
              }}
              disabled={isTranslating}
            />
          </div>

          <button
            type="submit"
            disabled={isTranslating || !originalText.trim()}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: isTranslating ? '#6c757d' : '#007cba',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: isTranslating ? 'not-allowed' : 'pointer'
            }}
          >
            {isTranslating ? 'Translating...' : 'Translate to Urdu'}
          </button>
        </form>

        {translatedText && (
          <div style={{ marginTop: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
              Translated Text
            </label>
            <div
              style={{
                width: '100%',
                padding: '0.5rem',
                border: '1px solid #ccc',
                borderRadius: '4px',
                minHeight: '120px',
                backgroundColor: '#f8f9fa',
                whiteSpace: 'pre-wrap'
              }}
            >
              {translatedText}
            </div>
          </div>
        )}

        <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '4px' }}>
          <h3>How Translation Works</h3>
          <p>
            This translation feature allows registered users to translate book content to Urdu
            while preserving technical terminology and code blocks. In the full implementation,
            this would connect to the backend API at <code>http://localhost:8000</code> to provide
            professional Urdu translation services using Google Gemini while maintaining
            technical accuracy.
          </p>
        </div>
      </div>
    </Layout>
  );
}