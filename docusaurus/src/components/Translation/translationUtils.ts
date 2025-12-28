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
    const response = await fetch('http://localhost:8001/api/v1/translate', {
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
  // For document pages, we need to replace the main content area
  const mainContent = document.querySelector('main div[class*="container"], main article, .markdown');
  if (mainContent) {
    // Store original content in a data attribute to allow toggling back
    if (!mainContent.hasAttribute('data-original-content')) {
      mainContent.setAttribute('data-original-content', mainContent.innerHTML);
    }

    // Create a temporary element to process the translated content
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = translatedContent;

    // Apply language-specific styling
    if (targetLanguage === 'ur') {
      mainContent.style.direction = 'rtl';
      mainContent.style.textAlign = 'right';
    } else {
      mainContent.style.direction = 'ltr';
      mainContent.style.textAlign = 'left';
    }

    // Update the content
    mainContent.innerHTML = tempDiv.innerHTML;
  } else {
    // If no main content found, fall back to the overlay method
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
          text-align: ${targetLanguage === 'ur' ? 'right' : 'left'};
        ">
          <h3 style="margin-top: 0;">Translated Content (${targetLanguage.toUpperCase()})</h3>
          <div id="translated-content" style="
            line-height: 1.6;
            margin: 15px 0;
          ">${translatedContent}</div>
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
  }
};

/**
 * Restore the original page content
 */
export const restoreOriginalContent = (): void => {
  const mainContent = document.querySelector('main div[class*="container"], main article, .markdown');
  if (mainContent && mainContent.hasAttribute('data-original-content')) {
    mainContent.innerHTML = mainContent.getAttribute('data-original-content') || '';
    mainContent.removeAttribute('data-original-content');
    mainContent.style.direction = 'ltr';
    mainContent.style.textAlign = 'left';
  }
};

/**
 * Translate a specific text string
 */
export const translateText = async (text: string, targetLanguage: string): Promise<string> => {
  try {
    const response = await fetch('http://localhost:8001/api/v1/translate', {
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