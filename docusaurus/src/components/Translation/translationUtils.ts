/**
 * Translation utilities for the Physical AI and Humanoid Robotics book platform
 */

interface TranslationResponse {
  original_content: string;
  translated_content: string;
  target_language: string;
}

/**
 * Translate the current page content using the backend API
 */
export const translatePageContent = async (targetLanguage: string): Promise<void> => {
  try {
    // Get the current page content
    const pageContent = getCurrentPageContent();

    if (!pageContent.trim()) {
      console.warn('No content to translate');
      return;
    }

    // Call the backend translation API
    const response = await fetch('/api/v1/translate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        content: pageContent,
        target_language: targetLanguage,
        source_language: 'en'
      }),
    });

    if (!response.ok) {
      throw new Error(`Translation API error: ${response.status} ${response.statusText}`);
    }

    const result: TranslationResponse = await response.json();

    // Update the page with translated content
    updatePageContent(result.translated_content, targetLanguage);

  } catch (error) {
    console.error('Translation error:', error);
    throw error;
  }
};

/**
 * Get the current page content that should be translated
 */
export const getCurrentPageContent = (): string => {
  // Get main content area
  const mainContent = document.querySelector('main, .main-wrapper, .container');
  if (!mainContent) {
    return '';
  }

  // Extract text content, trying to preserve structure
  const content = mainContent.textContent || '';

  // For more sophisticated extraction, we could also consider:
  // - Getting specific article/blog content
  // - Excluding navigation, headers, footers
  // - Preserving code blocks separately

  return content.substring(0, 4000); // Limit to first 4000 characters to avoid API limits
};

/**
 * Update the page with translated content
 */
export const updatePageContent = (translatedContent: string, targetLanguage: string): void => {
  // For a complete implementation, this would involve:
  // 1. Replacing text content with translations
  // 2. Handling RTL languages like Urdu
  // 3. Preserving code blocks and structure

  // Simple implementation: add a translation overlay
  const existingOverlay = document.getElementById('translation-overlay');
  if (existingOverlay) {
    existingOverlay.remove();
  }

  // Create a modal/overlay for the translated content
  const overlay = document.createElement('div');
  overlay.id = 'translation-overlay';
  overlay.innerHTML = `
    <div style="
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0,0,0,0.8);
      z-index: 10000;
      display: flex;
      justify-content: center;
      align-items: center;
    ">
      <div style="
        background: white;
        padding: 20px;
        border-radius: 8px;
        max-width: 90%;
        max-height: 90%;
        overflow-y: auto;
        direction: ${targetLanguage === 'ur' ? 'rtl' : 'ltr'};
      ">
        <h3 style="margin-top: 0;">Translated Content (${targetLanguage.toUpperCase()})</h3>
        <div id="translated-content">${translatedContent}</div>
        <button id="close-translation" style="
          margin-top: 10px;
          padding: 8px 16px;
          background: #8B5CF6;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        ">Close</button>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  // Add close functionality
  document.getElementById('close-translation')?.addEventListener('click', () => {
    overlay.remove();
  });
};

/**
 * Translate a specific text string
 */
export const translateText = async (text: string, targetLanguage: string): Promise<string> => {
  try {
    const response = await fetch('/api/v1/translate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        content: text,
        target_language: targetLanguage,
        source_language: 'en'
      }),
    });

    if (!response.ok) {
      throw new Error(`Translation API error: ${response.status} ${response.statusText}`);
    }

    const result: TranslationResponse = await response.json();
    return result.translated_content;
  } catch (error) {
    console.error('Text translation error:', error);
    // Return original text if translation fails
    return text;
  }
};