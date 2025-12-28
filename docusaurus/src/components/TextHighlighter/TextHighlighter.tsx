import React, { useEffect } from 'react';

const TextHighlighter = () => {
  useEffect(() => {
    // Initialize the text highlighter when the component mounts
    const initHighlighter = () => {
      let currentSelection: Selection | null = null;
      let toolbar: HTMLElement | null = null;
      let tooltip: HTMLElement | null = null;
      let hoveredHighlightId: string | null = null;

      // Create toolbar element
      const createToolbar = () => {
        const toolbarEl = document.createElement('div');
        toolbarEl.className = 'text-highlighter-toolbar';
        toolbarEl.style.cssText = `
          position: absolute;
          z-index: 10000;
          background-color: rgba(30, 41, 59, 0.9);
          border: 1px solid #8b5cf6;
          border-radius: 12px;
          padding: 8px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
          display: flex;
          align-items: center;
          gap: 4px;
          opacity: 0;
          transform: translateY(-10px);
          transition: opacity 0.2s ease, transform 0.2s ease;
          pointer-events: none;
        `;
        return toolbarEl;
      };

      // Create tooltip element
      const createTooltip = () => {
        const tooltipEl = document.createElement('div');
        tooltipEl.className = 'highlight-tooltip';
        tooltipEl.style.cssText = `
          position: absolute;
          z-index: 10001;
          background-color: rgba(30, 41, 59, 0.9);
          border: 1px solid #8b5cf6;
          border-radius: 12px;
          padding: 8px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
          font-size: 14px;
          opacity: 0;
          transform: translateY(-10px);
          transition: opacity 0.2s ease, transform 0.2s ease;
          pointer-events: none;
        `;
        return tooltipEl;
      };

      // Initialize toolbar and tooltip
      const toolbarElement = createToolbar();
      const tooltipElement = createTooltip();
      toolbar = toolbarElement;
      tooltip = tooltipElement;
      document.body.appendChild(toolbar);
      document.body.appendChild(tooltip);

      // Highlight colors
      const highlightColors = [
        { name: 'Yellow', value: 'rgba(254, 240, 138, 0.6)', darkValue: 'rgba(167, 139, 250, 0.4)' },
        { name: 'Green', value: 'rgba(162, 230, 138, 0.6)', darkValue: 'rgba(74, 222, 128, 0.4)' },
        { name: 'Blue', value: 'rgba(138, 180, 254, 0.6)', darkValue: 'rgba(96, 165, 250, 0.4)' },
        { name: 'Pink', value: 'rgba(254, 138, 203, 0.6)', darkValue: 'rgba(244, 114, 182, 0.4)' },
      ];

      // Get current page path
      const getCurrentPagePath = () => {
        return window.location.pathname;
      };

      // Get highlights from localStorage
      const getHighlights = () => {
        const pagePath = getCurrentPagePath();
        const key = `book_highlights_${pagePath}`;
        const saved = localStorage.getItem(key);
        return saved ? JSON.parse(saved) : [];
      };

      // Save highlights to localStorage
      const saveHighlights = (highlights: any[]) => {
        const pagePath = getCurrentPagePath();
        const key = `book_highlights_${pagePath}`;
        localStorage.setItem(key, JSON.stringify(highlights));
      };

      // Apply highlights to the DOM
      const applyHighlights = () => {
        // Remove existing highlights
        const existingHighlights = document.querySelectorAll('.book-highlight');
        existingHighlights.forEach(el => {
          const parent = el.parentElement;
          if (parent) {
            parent.replaceChild(document.createTextNode(el.textContent || ''), el);
          }
        });

        // Apply saved highlights
        const highlights = getHighlights();
        highlights.forEach((highlight: any) => {
          restoreHighlight(highlight);
        });
      };

      // Restore a single highlight
      const restoreHighlight = (highlight: any) => {
        // Find the text in the document
        const walker = document.createTreeWalker(
          document.body,
          NodeFilter.SHOW_TEXT,
          {
            acceptNode: function(node) {
              if (node.textContent && node.textContent?.includes(highlight.text)) {
                return NodeFilter.FILTER_ACCEPT;
              }
              return NodeFilter.FILTER_REJECT;
            }
          }
        );

        let node;
        while (node = walker.nextNode()) {
          if (node.textContent && node.textContent.includes(highlight.text)) {
            const text = node.textContent;
            const startIndex = text.indexOf(highlight.text);
            if (startIndex !== -1) {
              const range = document.createRange();
              range.setStart(node, startIndex);
              range.setEnd(node, startIndex + highlight.text.length);

              const highlightSpan = document.createElement('span');
              highlightSpan.className = 'book-highlight';
              highlightSpan.dataset.highlightId = highlight.id;
              highlightSpan.dataset.highlightText = highlight.text;

              const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
              highlightSpan.style.backgroundColor = isDarkMode
                ? highlightColors.find(c => c.name === highlight.color)?.darkValue || 'rgba(167, 139, 250, 0.4)'
                : highlightColors.find(c => c.name === highlight.color)?.value || 'rgba(254, 240, 138, 0.6)';

              highlightSpan.style.borderRadius = '4px';
              highlightSpan.style.padding = '0 2px';
              highlightSpan.style.cursor = 'pointer';

              range.surroundContents(highlightSpan);
              break;
            }
          }
        }
      };

      // Highlight selected text
      const highlightText = (color: string = 'Yellow') => {
        if (!currentSelection || currentSelection.toString().trim() === '') return;

        const selectedText = currentSelection.toString().trim();
        if (!selectedText) return;

        // Create a unique ID for this highlight
        const highlightId = `highlight-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

        // Get the range of the selection
        const range = currentSelection.getRangeAt(0);

        // Create highlight span
        const highlightSpan = document.createElement('span');
        highlightSpan.className = 'book-highlight';
        highlightSpan.dataset.highlightId = highlightId;
        highlightSpan.dataset.highlightText = selectedText;

        const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
        highlightSpan.style.backgroundColor = isDarkMode
          ? highlightColors.find(c => c.name === color)?.darkValue || 'rgba(167, 139, 250, 0.4)'
          : highlightColors.find(c => c.name === color)?.value || 'rgba(254, 240, 138, 0.6)';

        highlightSpan.style.borderRadius = '4px';
        highlightSpan.style.padding = '0 2px';
        highlightSpan.style.cursor = 'pointer';

        // Surround the selected content with the highlight span
        range.surroundContents(highlightSpan);

        // Save the highlight
        const highlights = getHighlights();
        const newHighlight = {
          id: highlightId,
          text: selectedText,
          color,
          pagePath: getCurrentPagePath(),
          timestamp: Date.now(),
        };
        highlights.push(newHighlight);
        saveHighlights(highlights);

        // Hide toolbar
        if (toolbar) {
          toolbar.style.opacity = '0';
          toolbar.style.pointerEvents = 'none';
        }

        currentSelection.removeAllRanges();
      };

      // Remove a highlight
      const removeHighlight = (id: string) => {
        const element = document.querySelector(`[data-highlight-id="${id}"]`) as HTMLElement;
        if (element && element.parentElement) {
          const parent = element.parentElement;
          // Get only the direct text content, excluding any child elements like tooltips
          let originalText = '';
          for (let i = 0; i < element.childNodes.length; i++) {
            const node = element.childNodes[i];
            if (node.nodeType === Node.TEXT_NODE) {
              originalText += node.textContent;
            }
          }
          parent.replaceChild(document.createTextNode(originalText), element);
        }

        const highlights = getHighlights();
        const updatedHighlights = highlights.filter((h: any) => h.id !== id);
        saveHighlights(updatedHighlights);

        if (tooltip) {
          tooltip.style.opacity = '0';
          tooltip.style.pointerEvents = 'none';
          hoveredHighlightId = null;
        }
      };

      // Show toolbar near selection
      const showToolbar = (rect: DOMRect) => {
        if (!toolbar) return;

        // Clear previous content
        toolbar.innerHTML = '';

        // Create highlight button
        const highlightBtn = document.createElement('button');
        highlightBtn.innerHTML = '🖍️';
        highlightBtn.title = 'Highlight text';
        highlightBtn.style.cssText = `
          background: #8b5cf6;
          color: white;
          border: 1px solid #8b5cf6;
          border-radius: 8px;
          padding: 6px 10px;
          cursor: pointer;
          font-size: 14px;
          display: flex;
          align-items: center;
          gap: 4px;
          transition: all 0.2s ease;
          box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3);
        `;

        // Add hover effect
        highlightBtn.onmouseenter = () => {
          highlightBtn.style.background = '#7c3aed';
          highlightBtn.style.boxShadow = '0 4px 12px rgba(139, 92, 246, 0.4)';
        };

        highlightBtn.onmouseleave = () => {
          highlightBtn.style.background = '#8b5cf6';
          highlightBtn.style.boxShadow = '0 2px 8px rgba(139, 92, 246, 0.3)';
        };
        highlightBtn.onclick = () => highlightText();
        toolbar.appendChild(highlightBtn);

        // Add color buttons
        highlightColors.forEach(color => {
          const colorBtn = document.createElement('button');
          colorBtn.style.cssText = `
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border: 2px solid #8b5cf6;
            background-color: ${document.documentElement.getAttribute('data-theme') === 'dark' ? color.darkValue : color.value};
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
          `;

          // Add hover effect for color buttons
          colorBtn.onmouseenter = () => {
            colorBtn.style.transform = 'scale(1.1)';
            colorBtn.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.3)';
          };

          colorBtn.onmouseleave = () => {
            colorBtn.style.transform = 'scale(1)';
            colorBtn.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.2)';
          };
          colorBtn.title = `Highlight with ${color.name}`;
          colorBtn.onclick = () => highlightText(color.name);
          toolbar.appendChild(colorBtn);
        });

        // Position toolbar above or below selection depending on space
        const top = rect.top + window.scrollY - 50;
        const left = rect.left + window.scrollX + rect.width / 2 - 50;

        toolbar.style.top = `${top}px`;
        toolbar.style.left = `${left}px`;
        toolbar.style.opacity = '1';
        toolbar.style.pointerEvents = 'auto';
      };

      // Handle text selection
      const handleSelection = () => {
        const selection = window.getSelection();
        if (!selection) return;

        const selectedText = selection.toString().trim();
        if (selectedText === '') {
          if (toolbar) {
            toolbar.style.opacity = '0';
            toolbar.style.pointerEvents = 'none';
          }
          currentSelection = null;
          return;
        }

        // Check if selection is inside a code block
        const range = selection.getRangeAt(0);
        const commonAncestor = range.commonAncestorContainer as Element;
        const codeElement = commonAncestor.closest('code, pre') || (commonAncestor.parentElement || commonAncestor).closest('code, pre');
        if (codeElement) {
          return;
        }

        // Check if selection is inside a highlighted area
        let isInsideHighlight = false;
        let parent = range.commonAncestorContainer as Element | null;
        while (parent) {
          if (parent.classList && parent.classList.contains('book-highlight')) {
            isInsideHighlight = true;
            break;
          }
          parent = parent.parentElement;
        }

        if (!isInsideHighlight) {
          const rect = range.getBoundingClientRect();
          currentSelection = selection;
          showToolbar(rect);
        }
      };

      // Handle clicks to hide toolbar
      const handleClick = (e: Event) => {
        const selection = window.getSelection();
        if (!selection || selection.toString().trim() === '') {
          if (toolbar) {
            toolbar.style.opacity = '0';
            toolbar.style.pointerEvents = 'none';
          }
        }
      };

      // Handle mouseup for desktop and touchend for mobile
      document.addEventListener('mouseup', handleSelection);
      document.addEventListener('touchend', handleSelection);
      document.addEventListener('click', handleClick);

      // Handle scroll to reposition toolbar
      const handleScroll = () => {
        if (currentSelection && toolbar) {
          const range = currentSelection.getRangeAt(0);
          const rect = range.getBoundingClientRect();
          const top = rect.top + window.scrollY - 50;
          const left = rect.left + window.scrollX + rect.width / 2 - 50;

          toolbar.style.top = `${top}px`;
          toolbar.style.left = `${left}px`;
        }
      };
      window.addEventListener('scroll', handleScroll);

      // Add event listeners for highlight interactions
      document.addEventListener('mouseover', (e) => {
        const target = e.target as HTMLElement;
        if (target.classList.contains('book-highlight')) {
          hoveredHighlightId = target.dataset.highlightId || null;

          if (tooltip && hoveredHighlightId) {
            // Clear previous content
            tooltip.innerHTML = '';

            // Create Ask AI button
            const askBtn = document.createElement('button');
            askBtn.textContent = 'Ask AI';
            askBtn.style.cssText = `
              background: #8b5cf6;
              color: white;
              border: 1px solid #8b5cf6;
              border-radius: 6px;
              padding: 6px 12px;
              margin-right: 8px;
              cursor: pointer;
              font-size: 12px;
              transition: all 0.2s ease;
              box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3);
            `;

            // Add hover effect for Ask AI button
            askBtn.onmouseenter = () => {
              askBtn.style.background = '#7c3aed';
              askBtn.style.boxShadow = '0 4px 12px rgba(139, 92, 246, 0.4)';
            };

            askBtn.onmouseleave = () => {
              askBtn.style.background = '#8b5cf6';
              askBtn.style.boxShadow = '0 2px 8px rgba(139, 92, 246, 0.3)';
            };
            askBtn.onclick = () => {
              // Trigger the AI chatbot with the selected text
              const event = new CustomEvent('askAIAboutText', { detail: { text: target.textContent } });
              document.dispatchEvent(event);
            };
            tooltip.appendChild(askBtn);

            // Create Remove button
            const removeBtn = document.createElement('button');
            removeBtn.innerHTML = '🗑️';
            removeBtn.style.cssText = `
              background: #e53e3e;
              color: white;
              border: 1px solid #e53e3e;
              border-radius: 6px;
              padding: 6px 12px;
              cursor: pointer;
              font-size: 12px;
              transition: all 0.2s ease;
              box-shadow: 0 2px 8px rgba(229, 62, 62, 0.3);
            `;

            // Add hover effect for Remove button
            removeBtn.onmouseenter = () => {
              removeBtn.style.background = '#c53030';
              removeBtn.style.boxShadow = '0 4px 12px rgba(229, 62, 62, 0.4)';
            };

            removeBtn.onmouseleave = () => {
              removeBtn.style.background = '#e53e3e';
              removeBtn.style.boxShadow = '0 2px 8px rgba(229, 62, 62, 0.3)';
            };
            removeBtn.onclick = () => {
              if (hoveredHighlightId) {
                removeHighlight(hoveredHighlightId);
              }
            };
            tooltip.appendChild(removeBtn);

            const rect = target.getBoundingClientRect();
            tooltip.style.top = `${rect.bottom + window.scrollY + 5}px`;
            tooltip.style.left = `${rect.left + window.scrollX}px`;
            tooltip.style.opacity = '1';
            tooltip.style.pointerEvents = 'auto';
          }
        }
      });

      document.addEventListener('mouseout', (e) => {
        const target = e.target as HTMLElement;
        if (target.classList.contains('book-highlight')) {
          if (tooltip) {
            tooltip.style.opacity = '0';
            tooltip.style.pointerEvents = 'none';
            hoveredHighlightId = null;
          }
        }
      });

      // Apply existing highlights on initialization
      applyHighlights();

      // Cleanup function
      return () => {
        document.removeEventListener('mouseup', handleSelection);
        document.removeEventListener('touchend', handleSelection);
        document.removeEventListener('click', handleClick);
        window.removeEventListener('scroll', handleScroll);
        document.removeEventListener('mouseover', () => {});
        document.removeEventListener('mouseout', () => {});

        if (toolbar && toolbar.parentNode) {
          toolbar.parentNode.removeChild(toolbar);
        }
        if (tooltip && tooltip.parentNode) {
          tooltip.parentNode.removeChild(tooltip);
        }
      };
    };

    // Initialize the highlighter when the DOM is ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initHighlighter);
    } else {
      initHighlighter();
    }
  }, []);

  return null;
};

export default TextHighlighter;