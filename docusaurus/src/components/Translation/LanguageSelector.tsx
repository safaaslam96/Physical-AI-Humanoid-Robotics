import React, { useState, useEffect } from 'react';
import { translatePageContent } from './translationUtils';

interface LanguageSelectorProps {
  currentLang?: string;
}

const LanguageSelector: React.FC<LanguageSelectorProps> = ({ currentLang = 'en' }) => {
  const [selectedLang, setSelectedLang] = useState<string>(currentLang);
  const [isTranslating, setIsTranslating] = useState<boolean>(false);
  const [translationStatus, setTranslationStatus] = useState<string>('');

  useEffect(() => {
    // Check for saved language preference
    const savedLang = localStorage.getItem('preferredLanguage') || 'en';
    setSelectedLang(savedLang);
  }, []);

  const handleLanguageChange = async (langCode: string) => {
    if (langCode === selectedLang) return; // No change needed

    setIsTranslating(true);
    setTranslationStatus('Translating content...');

    try {
      // Save preference
      localStorage.setItem('preferredLanguage', langCode);

      // Translate the page content
      await translatePageContent(langCode);

      setSelectedLang(langCode);
      setTranslationStatus('Translation complete!');

      // Hide status after 2 seconds
      setTimeout(() => setTranslationStatus(''), 2000);
    } catch (error) {
      console.error('Translation error:', error);
      setTranslationStatus('Translation failed. Please try again.');
    } finally {
      setIsTranslating(false);
    }
  };

  return (
    <div className="language-selector-container">
      <select
        value={selectedLang}
        onChange={(e) => handleLanguageChange(e.target.value)}
        disabled={isTranslating}
        className="language-selector-dropdown"
      >
        <option value="en">English</option>
        <option value="ur">اردو</option>
      </select>

      {isTranslating && (
        <div className="translation-loading">
          <span className="loading-spinner" />
          <span>Translating...</span>
        </div>
      )}

      {translationStatus && (
        <div className={`translation-status ${translationStatus.includes('failed') ? 'error' : 'success'}`}>
          {translationStatus}
        </div>
      )}
    </div>
  );
};

export default LanguageSelector;