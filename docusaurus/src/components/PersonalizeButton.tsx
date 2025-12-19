import React, { useState } from 'react';

const PersonalizeButton: React.FC = () => {
  const [isPersonalized, setIsPersonalized] = useState(false);
  const [loading, setLoading] = useState(false);

  const handlePersonalize = async () => {
    setLoading(true);

    // Check if user is logged in
    const token = localStorage.getItem('sessionToken');
    if (!token) {
      alert('Please sign in to access personalization features');
      setLoading(false);
      return;
    }

    try {
      // Get user background from localStorage (in real app, this would come from backend)
      const userBackground = localStorage.getItem('userBackground');

      if (userBackground) {
        // Call personalization skill
        const response = await fetch('/api/personalize', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            chapter_content: document.body.innerHTML, // In real app, this would be the actual chapter content
            user_background: JSON.parse(userBackground),
            chapter_metadata: {
              id: window.location.pathname,
              title: document.title,
            },
            locale: 'en',
          }),
        });

        if (response.ok) {
          const result = await response.json();
          // Update the page content with personalized version
          // In a real implementation, this would update the actual content
          setIsPersonalized(true);
        }
      } else {
        alert('Please update your profile with your technical background for personalization');
      }
    } catch (error) {
      console.error('Personalization error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginBottom: '1rem' }}>
      <button
        onClick={handlePersonalize}
        disabled={loading || isPersonalized}
        style={{
          backgroundColor: isPersonalized ? '#4CAF50' : '#800000',
          color: 'white',
          border: 'none',
          padding: '8px 16px',
          borderRadius: '4px',
          cursor: loading || isPersonalized ? 'not-allowed' : 'pointer',
          opacity: loading || isPersonalized ? 0.7 : 1,
        }}
      >
        {loading ? 'Personalizing...' : isPersonalized ? 'Personalized ✓' : 'Personalize Content'}
      </button>
      {isPersonalized && (
        <span style={{ marginLeft: '8px', fontSize: '0.9em', color: '#666' }}>
          Content adapted to your background
        </span>
      )}
    </div>
  );
};

export default PersonalizeButton;