import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import { translatePageContent, translateText } from '../components/Translation/translationUtils';
import styles from '../components/Translation/translation.module.css';

const TranslationPage = () => {
  const [sourceText, setSourceText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [targetLanguage, setTargetLanguage] = useState('ur');
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');

  const supportedLanguages = [
    { code: 'ur', name: 'Urdu' },
    { code: 'hi', name: 'Hindi' },
    { code: 'ar', name: 'Arabic' },
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'zh', name: 'Chinese' },
    { code: 'ja', name: 'Japanese' },
    { code: 'ko', name: 'Korean' }
  ];

  const handleTranslate = async () => {
    if (!sourceText.trim()) {
      setError('Please enter text to translate');
      return;
    }

    setIsLoading(true);
    setError('');
    setStatus('');

    try {
      const result = await translateText(sourceText, targetLanguage);
      setTranslatedText(result);
      setStatus('Translation completed successfully!');
    } catch (err) {
      console.error('Translation error:', err);
      setError(`Translation failed: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePageTranslate = async () => {
    setIsLoading(true);
    setError('');
    setStatus('');

    try {
      await translatePageContent(targetLanguage);
      setStatus('Page translated successfully!');
    } catch (err) {
      console.error('Page translation error:', err);
      setError(`Page translation failed: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout title="Translation" description="Translate content to different languages">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>Content Translation</h1>
            <p className="margin-bottom--lg">
              Translate text or entire pages to different languages, including Urdu for better accessibility.
            </p>

            <div className={styles.languageSelectorContainer}>
              <label htmlFor="target-language" className="margin-right--md">
                Target Language:
              </label>
              <select
                id="target-language"
                className={styles.languageSelectorDropdown}
                value={targetLanguage}
                onChange={(e) => setTargetLanguage(e.target.value)}
              >
                {supportedLanguages.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.name} ({lang.code.toUpperCase()})
                  </option>
                ))}
              </select>
            </div>

            <div className="margin-vert--lg">
              <div className="row">
                <div className="col col--6">
                  <h3>Source Text</h3>
                  <textarea
                    value={sourceText}
                    onChange={(e) => setSourceText(e.target.value)}
                    placeholder="Enter text to translate..."
                    rows={8}
                    style={{
                      width: '100%',
                      padding: '12px',
                      border: '1px solid var(--ifm-color-emphasis-300)',
                      borderRadius: '6px',
                      fontSize: '16px',
                      fontFamily: 'inherit'
                    }}
                  />
                </div>
                <div className="col col--6">
                  <h3>Translated Text</h3>
                  <div
                    style={{
                      width: '100%',
                      padding: '12px',
                      border: '1px solid var(--ifm-color-emphasis-300)',
                      borderRadius: '6px',
                      minHeight: '120px',
                      backgroundColor: 'var(--ifm-color-emphasis-100)',
                      fontSize: '16px',
                      fontFamily: 'inherit',
                      direction: targetLanguage === 'ur' ? 'rtl' : 'ltr',
                      textAlign: targetLanguage === 'ur' ? 'right' : 'left'
                    }}
                  >
                    {isLoading ? (
                      <div className={styles.translationLoading}>
                        <div className={styles.loadingSpinner}></div>
                        <span>Translating...</span>
                      </div>
                    ) : translatedText ? (
                      translatedText
                    ) : (
                      <span className="text--secondary">Translation will appear here...</span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="button-group margin-vert--md">
              <button
                className="button button--primary"
                onClick={handleTranslate}
                disabled={isLoading}
              >
                {isLoading ? (
                  <div className={styles.translationLoading}>
                    <div className={styles.loadingSpinner}></div>
                    <span>Translating...</span>
                  </div>
                ) : (
                  'Translate Text'
                )}
              </button>

              <button
                className="button button--secondary"
                onClick={handlePageTranslate}
                disabled={isLoading}
              >
                Translate Current Page
              </button>
            </div>

            {(status || error) && (
              <div className={`${styles.translationStatus} ${error ? styles.error : styles.success}`}>
                {error || status}
              </div>
            )}

            <div className="margin-vert--lg">
              <h3>How to Use</h3>
              <ul>
                <li>Enter text in the source text area</li>
                <li>Select your target language (Urdu is fully supported)</li>
                <li>Click "Translate Text" to see the translation</li>
                <li>Use "Translate Current Page" to translate the entire current page</li>
              </ul>

              <h3 className="margin-top--lg">Urdu Translation Features</h3>
              <ul>
                <li>Proper RTL (Right-to-Left) text direction</li>
                <li>Accurate technical terminology preservation</li>
                <li>Context-aware translations for robotics and AI content</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default TranslationPage;