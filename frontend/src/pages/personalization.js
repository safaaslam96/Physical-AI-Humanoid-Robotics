import React from 'react';
import Layout from '@theme/Layout';
import { useState } from 'react';

export default function PersonalizationPage() {
  const [profile, setProfile] = useState({
    softwareBackground: '',
    hardwareBackground: '',
    learningGoals: ''
  });
  const [isSaved, setIsSaved] = useState(false);

  const handleChange = (e) => {
    setProfile({
      ...profile,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // In a real implementation, this would send data to the backend
    console.log('Profile submitted:', profile);
    setIsSaved(true);
    setTimeout(() => setIsSaved(false), 3000);
  };

  return (
    <Layout title="Personalize Your Learning" description="Personalize your learning experience based on your background and goals">
      <div style={{
        maxWidth: '800px',
        margin: '0 auto',
        padding: '2rem'
      }}>
        <h1>Personalize Your Learning Experience</h1>
        <p>Tell us about your background and goals to get personalized content recommendations.</p>

        <form onSubmit={handleSubmit} style={{ marginTop: '1rem' }}>
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
              Software Background
            </label>
            <textarea
              name="softwareBackground"
              value={profile.softwareBackground}
              onChange={handleChange}
              placeholder="e.g., Python, C++, Machine Learning, ROS, etc."
              style={{
                width: '100%',
                padding: '0.5rem',
                border: '1px solid #ccc',
                borderRadius: '4px',
                minHeight: '80px'
              }}
            />
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
              Hardware/Robotics Background
            </label>
            <textarea
              name="hardwareBackground"
              value={profile.hardwareBackground}
              onChange={handleChange}
              placeholder="e.g., Electronics, Control Systems, Mechanical Design, etc."
              style={{
                width: '100%',
                padding: '0.5rem',
                border: '1px solid #ccc',
                borderRadius: '4px',
                minHeight: '80px'
              }}
            />
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
              Learning Goals
            </label>
            <textarea
              name="learningGoals"
              value={profile.learningGoals}
              onChange={handleChange}
              placeholder="e.g., Build humanoid robots, Understand AI-robotics integration, etc."
              style={{
                width: '100%',
                padding: '0.5rem',
                border: '1px solid #ccc',
                borderRadius: '4px',
                minHeight: '80px'
              }}
            />
          </div>

          <button
            type="submit"
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Save Profile
          </button>

          {isSaved && (
            <div style={{
              marginTop: '1rem',
              padding: '0.5rem',
              backgroundColor: '#d4edda',
              color: '#155724',
              borderRadius: '4px'
            }}>
              Profile saved successfully! Your content will now be personalized based on your background.
            </div>
          )}
        </form>

        <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#f8f9fa', borderRadius: '4px' }}>
          <h3>How Personalization Works</h3>
          <p>
            Once you save your profile, the system will adapt explanations, examples, and difficulty levels
            to match your background and learning goals. In the full implementation, this would connect
            to the backend API at <code>http://localhost:8000</code> to store your preferences and
            retrieve personalized content.
          </p>
        </div>
      </div>
    </Layout>
  );
}