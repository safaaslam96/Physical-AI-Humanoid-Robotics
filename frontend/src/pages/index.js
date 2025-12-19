import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div
          className={styles.fadeIn}
          style={{
            opacity: isVisible ? 1 : 0,
            transform: isVisible ? 'translateY(0)' : 'translateY(20px)',
            transition: 'opacity 0.8s ease, transform 0.8s ease',
            transitionDelay: '0.1s'
          }}
        >
          {/* Logo Section */}
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            marginBottom: '2rem'
          }}>
            <img
              src="/Physical-AI-Humanoid-Robotics/img/pngtree-cute-robot-mascot-logo-png-image_8493119.png"
              alt="Physical AI & Humanoid Robotics Logo"
              style={{
                width: '150px',
                height: '150px',
                borderRadius: '20px',
                boxShadow: '0 10px 30px rgba(111, 66, 193, 0.3)',
                transition: 'transform 0.3s ease, box-shadow 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.target.style.transform = 'scale(1.1) rotate(5deg)';
                e.target.style.boxShadow = '0 15px 40px rgba(111, 66, 193, 0.4)';
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'scale(1) rotate(0deg)';
                e.target.style.boxShadow = '0 10px 30px rgba(111, 66, 193, 0.3)';
              }}
            />
          </div>

          <h1 className="hero__title" style={{
            background: 'linear-gradient(135deg, #0a1f44, #1b3b6f, #6f42c1)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            fontSize: '3rem',
            fontWeight: 'bold',
            marginBottom: '1rem',
            textAlign: 'center'
          }}>
            {siteConfig.title}
          </h1>

          <p className="hero__subtitle" style={{
            fontSize: '1.5rem',
            color: '#495057',
            marginBottom: '2rem',
            textAlign: 'center'
          }}>
            {siteConfig.tagline}
          </p>

          <div className={styles.buttons} style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '1rem',
            flexWrap: 'wrap'
          }}>
            <Link
              className="button button--secondary button--lg"
              style={{
                backgroundColor: '#0a1f44',
                borderColor: '#0a1f44',
                fontSize: '1.1rem',
                padding: '0.75rem 1.5rem',
                transition: 'all 0.3s ease',
                boxShadow: '0 4px 15px rgba(10, 31, 68, 0.2)',
                borderRadius: '12px'
              }}
              onMouseEnter={(e) => {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = '0 6px 20px rgba(10, 31, 68, 0.3)';
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = '0 4px 15px rgba(10, 31, 68, 0.2)';
              }}
              to="/Physical-AI-Humanoid-Robotics/docs/intro">
              Read the Book - 5min ⏱️
            </Link>

            <Link
              className="button button--primary button--lg"
              style={{
                backgroundColor: '#6f42c1',
                borderColor: '#6f42c1',
                fontSize: '1.1rem',
                padding: '0.75rem 1.5rem',
                transition: 'all 0.3s ease',
                boxShadow: '0 4px 15px rgba(111, 66, 193, 0.2)',
                borderRadius: '12px'
              }}
              onMouseEnter={(e) => {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = '0 6px 20px rgba(111, 66, 193, 0.3)';
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = '0 4px 15px rgba(111, 66, 193, 0.2)';
              }}
              to="/Physical-AI-Humanoid-Robotics/docs/logo-showcase">
              View Our Logo
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="AI Systems in the Physical World. Embodied Intelligence.">
      <HomepageHeader />
      <main>
        <div
          style={{
            opacity: isLoaded ? 1 : 0,
            transform: isLoaded ? 'translateY(0)' : 'translateY(30px)',
            transition: 'opacity 0.8s ease, transform 0.8s ease',
            transitionDelay: '0.3s'
          }}
        >
          <HomepageFeatures />
        </div>
      </main>
    </Layout>
  );
}