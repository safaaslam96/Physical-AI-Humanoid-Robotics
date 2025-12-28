function initializeHighlighter() {
  let isHighlighting = false;
  let highlightColor = 'yellow';

  // Create highlighter toggle button
  const highlighterBtn = document.createElement('button');
  highlighterBtn.id = 'highlighter-toggle';
  highlighterBtn.innerHTML = '✏️';
  highlighterBtn.title = 'Highlight Text';
  highlighterBtn.className = 'highlighter-toggle-button';

  // Add highlighter button to page
  document.body.appendChild(highlighterBtn);

  // Toggle highlighter mode
  highlighterBtn.addEventListener('click', function() {
    isHighlighting = !isHighlighting;
    this.classList.toggle('highlighter-active', isHighlighting);

    if (isHighlighting) {
      document.body.style.cursor = 'crosshair';
    } else {
      document.body.style.cursor = 'default';
    }
  });

  // Handle text selection and highlighting
  document.addEventListener('mouseup', function() {
    if (!isHighlighting) return;

    const selection = window.getSelection();
    if (!selection.toString().trim()) return;

    const range = selection.getRangeAt(0);
    if (!range) return;

    // Don't highlight in code blocks or other restricted areas
    if (range.commonAncestorContainer.nodeType !== Node.TEXT_NODE) {
      const parentElement = range.commonAncestorContainer.parentElement;
      if (parentElement.closest('pre, code, .code-block, .prism-code, .navbar, .menu')) {
        return;
      }
    } else {
      const parentElement = range.commonAncestorContainer.parentElement;
      if (parentElement.closest('pre, code, .code-block, .prism-code, .navbar, .menu')) {
        return;
      }
    }

    // Create highlight element
    const highlight = document.createElement('mark');
    highlight.className = 'text-highlight';
    highlight.setAttribute('data-highlight', 'true');

    // Add tooltip functionality
    highlight.addEventListener('mouseenter', function() {
      const tooltip = document.createElement('div');
      tooltip.className = 'highlight-tooltip';
      tooltip.textContent = 'Click to remove highlight';
      this.appendChild(tooltip);
    });

    highlight.addEventListener('mouseleave', function() {
      const tooltip = this.querySelector('.highlight-tooltip');
      if (tooltip) tooltip.remove();
    });

    highlight.addEventListener('click', function() {
      // Remove highlight and restore text
      const parent = this.parentNode;
      const textNode = document.createTextNode(this.textContent);
      parent.replaceChild(textNode, this);
      parent.normalize();
    });

    try {
      range.surroundContents(highlight);
      selection.removeAllRanges();
    } catch (e) {
      console.log('Cannot highlight this element');
    }
  });
}

// Initialize highlighter when page loads
document.addEventListener('DOMContentLoaded', function() {
  initializeHighlighter();
});

// Make functions available globally
window.PhysicalAI = window.PhysicalAI || {};