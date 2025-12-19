/**
 * Global translation functionality for the Physical AI and Humanoid Robotics book platform
 */

// Translation state
let currentLanguage = 'en';
let isTranslating = false;

// Language configuration
const languageConfig = {
  'en': {
    name: 'English',
    direction: 'ltr'
  },
  'ur': {
    name: 'Urdu',
    direction: 'rtl'
  }
};

// Initialize translation functionality when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  initializeTranslationUI();
  setupLanguageSelectors();
  initializeChatbot();
});

/**
 * Initialize translation UI elements
 */
function initializeTranslationUI() {
  // Add event listeners to any language selector dropdowns
  const languageSelectors = document.querySelectorAll('.language-selector');
  languageSelectors.forEach(selector => {
    selector.addEventListener('click', function(e) {
      e.preventDefault();
      const lang = this.getAttribute('data-lang');
      if (lang) {
        changePageLanguage(lang);
      }
    });
  });

  // Add click handler to the main translation dropdown button
  const translateButtons = document.querySelectorAll('[data-language-toggle]');
  translateButtons.forEach(button => {
    button.addEventListener('click', function(e) {
      e.preventDefault();
      // Toggle between English and Urdu
      const newLang = currentLanguage === 'en' ? 'ur' : 'en';
      changePageLanguage(newLang);
    });
  });
}

/**
 * Set up language selector elements
 */
function setupLanguageSelectors() {
  // Look for any elements with language-selector class
  const selectors = document.querySelectorAll('.language-selector');
  selectors.forEach(selector => {
    // Already handled in initializeTranslationUI, but keeping for compatibility
    if (!selector.hasAttribute('data-initialized')) {
      selector.addEventListener('click', function(e) {
        e.preventDefault();
        const lang = this.getAttribute('data-lang');
        if (lang && lang !== currentLanguage) {
          changePageLanguage(lang);
        }
      });
      selector.setAttribute('data-initialized', 'true');
    }
  });
}

/**
 * Change the language of the current page
 */
async function changePageLanguage(targetLanguage) {
  if (isTranslating || targetLanguage === currentLanguage) {
    return;
  }

  isTranslating = true;
  const originalContent = document.querySelector('main') || document.body;
  const loadingIndicator = showLoadingIndicator();

  try {
    // Save the original content if not already saved
    if (!originalContent.getAttribute('data-original-content')) {
      originalContent.setAttribute('data-original-content', originalContent.innerHTML);
    }

    // Get page text for translation
    const pageText = extractPageText(originalContent);

    // Call the backend API to translate content
    const response = await fetch('/api/v1/translate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        content: pageText,
        target_language: targetLanguage,
        source_language: currentLanguage
      })
    });

    if (!response.ok) {
      throw new Error(`Translation API error: ${response.status}`);
    }

    const result = await response.json();

    // Apply translated content to the page
    applyTranslation(result.translated_content, targetLanguage);

    // Update current language
    currentLanguage = targetLanguage;

    // Update UI to reflect new language
    updateLanguageUI(targetLanguage);

    // Hide loading indicator
    hideLoadingIndicator(loadingIndicator);

    console.log(`Page translated to ${languageConfig[targetLanguage]?.name || targetLanguage}`);
  } catch (error) {
    console.error('Translation error:', error);
    hideLoadingIndicator(loadingIndicator);
    showTranslationError(error.message);
  } finally {
    isTranslating = false;
  }
}

/**
 * Extract text content from the page for translation
 */
function extractPageText(element) {
  // Get all text nodes and important content
  let textContent = '';

  // Get main content areas
  const mainContent = element.querySelector('main, .main-wrapper, .container') || element;

  // Extract text while preserving structure
  const walker = document.createTreeWalker(
    mainContent,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode: function(node) {
        // Only include text nodes that are visible and not in certain elements
        const parentTag = node.parentElement.tagName.toLowerCase();
        if (['script', 'style', 'noscript', 'meta', 'link', 'title'].includes(parentTag)) {
          return NodeFilter.FILTER_REJECT;
        }

        // Only include text nodes with actual content
        if (node.textContent.trim().length > 0) {
          return NodeFilter.FILTER_ACCEPT;
        }

        return NodeFilter.FILTER_REJECT;
      }
    }
  );

  const textNodes = [];
  let node;
  while (node = walker.nextNode()) {
    textNodes.push(node.textContent.trim());
  }

  // Combine text nodes, limiting to reasonable size for API
  textContent = textNodes.filter(text => text.length > 0).join(' ').substring(0, 3000);

  return textContent;
}

/**
 * Apply translated content to the page
 */
function applyTranslation(translatedContent, targetLanguage) {
  const mainContent = document.querySelector('main') || document.body;

  // For a real implementation, we would map translated segments back to original elements
  // For now, we'll show a translation overlay

  // Store original content if not already stored
  if (!mainContent.getAttribute('data-original-content')) {
    mainContent.setAttribute('data-original-content', mainContent.innerHTML);
  }

  // Create translation overlay
  const overlay = document.createElement('div');
  overlay.id = 'translation-overlay';
  overlay.style.cssText = `
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
  `;

  const contentBox = document.createElement('div');
  contentBox.style.cssText = `
    background: white;
    padding: 20px;
    border-radius: 8px;
    max-width: 90%;
    max-height: 90%;
    overflow-y: auto;
    direction: ${languageConfig[targetLanguage]?.direction || 'ltr'};
  `;

  const header = document.createElement('h3');
  header.textContent = `Translation to ${languageConfig[targetLanguage]?.name || targetLanguage.toUpperCase()}`;
  header.style.cssText = 'margin-top: 0; color: #333;';

  const translatedText = document.createElement('div');
  translatedText.id = 'translated-text';
  translatedText.textContent = translatedContent;
  translatedText.style.cssText = 'margin: 15px 0; line-height: 1.6;';

  const buttonContainer = document.createElement('div');
  buttonContainer.style.cssText = 'margin-top: 15px; text-align: right;';

  const restoreButton = document.createElement('button');
  restoreButton.textContent = 'Restore Original';
  restoreButton.style.cssText = `
    padding: 8px 16px;
    background: #8B5CF6;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    margin-right: 10px;
  `;
  restoreButton.onclick = function() {
    document.body.removeChild(overlay);
  };

  const closeButton = document.createElement('button');
  closeButton.textContent = 'Close';
  closeButton.style.cssText = `
    padding: 8px 16px;
    background: #6B7280;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  `;
  closeButton.onclick = function() {
    document.body.removeChild(overlay);
  };

  buttonContainer.appendChild(restoreButton);
  buttonContainer.appendChild(closeButton);

  contentBox.appendChild(header);
  contentBox.appendChild(translatedText);
  contentBox.appendChild(buttonContainer);
  overlay.appendChild(contentBox);

  document.body.appendChild(overlay);
}

/**
 * Update UI elements to reflect current language
 */
function updateLanguageUI(targetLanguage) {
  // Update any language indicator in the UI
  const langIndicators = document.querySelectorAll('[data-current-lang]');
  langIndicators.forEach(indicator => {
    indicator.textContent = languageConfig[targetLanguage]?.name || targetLanguage;
  });

  // Update document direction for RTL languages
  document.documentElement.dir = languageConfig[targetLanguage]?.direction || 'ltr';

  // Store preference in localStorage
  localStorage.setItem('preferredLanguage', targetLanguage);
}

/**
 * Show loading indicator
 */
function showLoadingIndicator() {
  const loadingDiv = document.createElement('div');
  loadingDiv.id = 'translation-loading';
  loadingDiv.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0,0,0,0.7);
    color: white;
    padding: 15px 25px;
    border-radius: 8px;
    z-index: 10001;
    display: flex;
    align-items: center;
    gap: 10px;
  `;

  const spinner = document.createElement('div');
  spinner.style.cssText = `
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255,255,255,0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  `;

  const style = document.createElement('style');
  style.textContent = `
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  `;
  document.head.appendChild(style);

  const text = document.createElement('span');
  text.textContent = 'Translating...';

  loadingDiv.appendChild(spinner);
  loadingDiv.appendChild(text);

  document.body.appendChild(loadingDiv);

  return loadingDiv;
}

/**
 * Hide loading indicator
 */
function hideLoadingIndicator(loadingElement) {
  if (loadingElement && loadingElement.parentNode) {
    loadingElement.parentNode.removeChild(loadingElement);
  }
}

/**
 * Show translation error
 */
function showTranslationError(message) {
  const errorDiv = document.createElement('div');
  errorDiv.id = 'translation-error';
  errorDiv.textContent = `Translation Error: ${message}`;
  errorDiv.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: #ef4444;
    color: white;
    padding: 15px 20px;
    border-radius: 6px;
    z-index: 10002;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  `;

  document.body.appendChild(errorDiv);

  // Remove error message after 5 seconds
  setTimeout(() => {
    if (errorDiv.parentNode) {
      errorDiv.parentNode.removeChild(errorDiv);
    }
  }, 5000);
}

/**
 * Initialize chatbot functionality
 */
function initializeChatbot() {
  // Set up the chatbot toggle button if it exists
  const chatbotToggle = document.getElementById('chatbot-toggle');
  if (chatbotToggle) {
    // The chatbot component will handle its own toggle functionality
    console.log('Chatbot toggle button initialized');
  }

  // Additional chatbot initialization can go here if needed
}

/**
 * Get the current preferred language
 */
function getPreferredLanguage() {
  return localStorage.getItem('preferredLanguage') || 'en';
}

/**
 * Restore original content
 */
function restoreOriginalContent() {
  const mainContent = document.querySelector('main') || document.body;
  const originalContent = mainContent.getAttribute('data-original-content');

  if (originalContent) {
    mainContent.innerHTML = originalContent;
    mainContent.removeAttribute('data-original-content');
  }
}

// Make functions available globally if needed
window.PhysicalAI = window.PhysicalAI || {};
window.PhysicalAI.translatePage = changePageLanguage;
window.PhysicalAI.getPreferredLanguage = getPreferredLanguage;
window.PhysicalAI.restoreOriginalContent = restoreOriginalContent;