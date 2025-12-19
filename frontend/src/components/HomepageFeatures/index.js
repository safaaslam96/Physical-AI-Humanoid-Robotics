import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'AI-Powered Learning',
    Svg: null,
    description: (
      <>
        Interactive book with embedded RAG chatbot that answers questions based on the content.
        Get instant help and explanations tailored to your learning style.
      </>
    ),
  },
  {
    title: 'Personalized Experience',
    Svg: null,
    description: (
      <>
        Content adapts to your background and learning goals. Tell us about your experience
        level and interests to get personalized explanations and examples.
      </>
    ),
  },
  {
    title: 'Multilingual Support',
    Svg: null,
    description: (
      <>
        Translate content to Urdu for enhanced accessibility. Professional translations
        that preserve technical accuracy and code examples.
      </>
    ),
  },
];

const ModuleList = [
  {
    title: 'Module 1: The Robotic Nervous System',
    description: 'ROS 2, Nodes, Topics, Services, and Python Agent integration',
    weeks: 'Weeks 3-5'
  },
  {
    title: 'Module 2: The Digital Twin',
    description: 'Gazebo physics simulation, Unity visualization, sensor systems',
    weeks: 'Weeks 6-7'
  },
  {
    title: 'Module 3: The AI-Robot Brain',
    description: 'NVIDIA Isaac™, Isaac Sim, synthetic data, VSLAM, Nav2',
    weeks: 'Weeks 8-10'
  },
  {
    title: 'Module 4: Vision-Language-Action & Capstone',
    description: 'Google Gemini, cognitive planning, autonomous humanoid tasks',
    weeks: 'Week 13'
  },
  {
    title: 'Introduction: Physical AI Fundamentals',
    description: 'Foundations of Physical AI, embodied intelligence, sensor systems',
    weeks: 'Weeks 1-2'
  },
  {
    title: 'Humanoid Robot Development',
    description: 'Kinematics, dynamics, bipedal locomotion, human-robot interaction',
    weeks: 'Weeks 11-12'
  }
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

function ModuleCard({ title, description, weeks }) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <div style={{
          backgroundColor: '#f8f9fa',
          padding: '1.5rem',
          borderRadius: '8px',
          borderLeft: '4px solid #007cba',
          transition: 'transform 0.2s ease, box-shadow 0.2s ease',
          height: '100%'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = 'translateY(-5px)';
          e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'translateY(0)';
          e.currentTarget.style.boxShadow = 'none';
        }}>
          <h3 style={{ color: '#007cba' }}>{title}</h3>
          <p>{description}</p>
          <div style={{
            backgroundColor: '#e9ecef',
            padding: '0.25rem 0.5rem',
            borderRadius: '12px',
            display: 'inline-block',
            fontSize: '0.8rem',
            marginTop: '0.5rem'
          }}>
            {weeks}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <>
      {/* Core Features Section */}
      <section className={styles.features}>
        <div className="container">
          <div className="row">
            {FeatureList.map((props, idx) => (
              <Feature key={idx} {...props} />
            ))}
          </div>
        </div>
      </section>

      {/* Modules Overview Section */}
      <section className={styles.modules}>
        <div className="container">
          <div className="text--center padding-horiz--md" style={{ marginBottom: '2rem' }}>
            <h2 style={{
              fontSize: '2rem',
              fontWeight: 'bold',
              background: 'linear-gradient(135deg, #007cba, #6f42c1)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text'
            }}>
              Course Modules Overview
            </h2>
            <p style={{ fontSize: '1.1rem', color: '#666' }}>
              A comprehensive 13-week journey through Physical AI and Humanoid Robotics
            </p>
          </div>

          <div className="row">
            {ModuleList.map((module, idx) => (
              <ModuleCard key={idx} {...module} />
            ))}
          </div>
        </div>
      </section>
    </>
  );
}