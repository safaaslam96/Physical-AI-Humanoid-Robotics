import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import styles from './styles.module.css';

const PersonalizationButton = ({ children }) => {
  const { user, isAuthenticated } = useAuth();
  const [isPersonalized, setIsPersonalized] = useState(false);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  const togglePersonalization = () => {
    if (isAuthenticated) {
      setIsPersonalized(!isPersonalized);
    } else {
      // Redirect to sign in if not authenticated
      window.location.href = '/signin';
    }
  };

  if (!isClient) {
    // Server-side rendering fallback
    return <div className={styles.personalizationPlaceholder}>{children}</div>;
  }

  // Check if user is authenticated and has profile data
  const canPersonalize = isAuthenticated && user &&
    (user.softwareBackground || user.hasHighEndGPU !== undefined || user.familiarWithROS2 !== undefined);

  if (!canPersonalize) {
    return (
      <div className={styles['personalization-container']}>
        <div className={styles['chapter-content']}>
          {children}
        </div>
      </div>
    );
  }

  return (
    <div className={styles['personalization-container']}>
      <div className={styles['personalization-controls']}>
        <button
          className={`${styles['personalization-toggle-button']} ${isPersonalized ? styles.active : ''}`}
          onClick={togglePersonalization}
        >
          {isPersonalized ? 'Show General View' : 'Personalize this chapter'}
        </button>
      </div>

      <div className={styles['chapter-content']}>
        {isPersonalized && isAuthenticated ? (
          <PersonalizedContent user={user} originalContent={children} />
        ) : (
          children
        )}
      </div>
    </div>
  );
};

const PersonalizedContent = ({ user, originalContent }) => {
  // Personalization logic based on user profile
  const userBackground = user?.softwareBackground || 'Beginner';
  const hasGPU = user?.hasHighEndGPU || false;
  const familiarWithROS2 = user?.familiarWithROS2 || false;

  // This component will wrap the original content with personalized elements
  // In a real implementation, this would involve more complex logic
  // to transform the content based on user preferences

  const getPersonalizationNote = () => {
    let notes = [];

    // Software background personalization
    switch(userBackground?.toLowerCase()) {
      case 'beginner':
        notes.push('Content adapted for beginners with simplified explanations');
        break;
      case 'intermediate':
        notes.push('Content balanced for intermediate users');
        break;
      case 'advanced':
        notes.push('Content includes advanced technical details and tips');
        break;
      default:
        notes.push('Content adapted for your background');
    }

    // GPU personalization
    if (hasGPU) {
      notes.push('NVIDIA Isaac Sim specific instructions included');
    } else {
      notes.push('Alternative instructions for cloud/lower-end GPU provided');
    }

    // ROS2 familiarity personalization
    if (!familiarWithROS2) {
      notes.push('Extra ROS2 introduction material provided');
    } else {
      notes.push('ROS2 content tailored for experienced users');
    }

    return notes;
  };

  // Enhanced personalization: create context-aware content adjustments
  const getPersonalizedSections = () => {
    const sections = [];

    // Add beginner/advanced content adjustments
    if (userBackground?.toLowerCase() === 'beginner') {
      sections.push(
        <div key="beginner-content" className="personalized-beginner-content">
          <details style={{border: '1px solid #e0e0e0', borderRadius: '4px', margin: '1rem 0', padding: '1rem', backgroundColor: '#f9f9f9'}}>
            <summary style={{fontWeight: 'bold', cursor: 'pointer'}}>💡 For Beginners: Additional Context</summary>
            <p style={{marginTop: '0.5rem'}}>This section may contain advanced concepts. Here's a simplified explanation to help you get started...</p>
          </details>
        </div>
      );
    } else if (userBackground?.toLowerCase() === 'advanced') {
      sections.push(
        <div key="advanced-content" className="personalized-advanced-content">
          <details style={{border: '1px solid #e0e0e0', borderRadius: '4px', margin: '1rem 0', padding: '1rem', backgroundColor: '#f0f8ff'}}>
            <summary style={{fontWeight: 'bold', cursor: 'pointer'}}>🔬 For Experts: Advanced Tips</summary>
            <p style={{marginTop: '0.5rem'}}>As an advanced user, you might be interested in these optimization techniques and advanced configurations...</p>
          </details>
        </div>
      );
    }

    // Add GPU-specific content
    if (hasGPU) {
      sections.push(
        <div key="gpu-content" className="personalized-gpu-content">
          <div style={{border: '1px solid #4caf50', borderRadius: '4px', margin: '1rem 0', padding: '1rem', backgroundColor: '#e8f5e8'}}>
            <strong>⚡ GPU Acceleration:</strong> Since you have a high-end GPU, you can enable advanced simulation features and run more complex models in real-time.
          </div>
        </div>
      );
    } else {
      sections.push(
        <div key="cloud-content" className="personalized-cloud-content">
          <div style={{border: '1px solid #2196f3', borderRadius: '4px', margin: '1rem 0', padding: '1rem', backgroundColor: '#e3f2fd'}}>
            <strong>☁️ Cloud Alternative:</strong> Since you don't have a high-end GPU, consider using cloud-based simulation services or running simplified models to avoid performance issues.
          </div>
        </div>
      );
    }

    // Add ROS2 familiarity content
    if (!familiarWithROS2) {
      sections.push(
        <div key="ros2-intro" className="personalized-ros2-intro">
          <details style={{border: '1px solid #ff9800', borderRadius: '4px', margin: '1rem 0', padding: '1rem', backgroundColor: '#fff3e0'}}>
            <summary style={{fontWeight: 'bold', cursor: 'pointer'}}>🤖 ROS 2 Introduction</summary>
            <p style={{marginTop: '0.5rem'}}>ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides services like hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more.</p>
          </details>
        </div>
      );
    }

    return sections;
  };

  return (
    <div>
      <div className={styles['personalization-note']}>
        <strong>Personalized for you:</strong><br />
        {getPersonalizationNote().map((note, index) => (
          <span key={index}>• {note}<br /></span>
        ))}
      </div>
      {getPersonalizedSections()}
      {originalContent}
    </div>
  );
};

export default PersonalizationButton;