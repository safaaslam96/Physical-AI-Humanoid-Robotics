import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import ChatbotPopup from '@site/src/components/Chatbot/ChatbotPopup';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="row">
          {/* Left side: Cover image */}
          <div className="col col--6">
            <img
              src="/static/img/physical-ai-humanoid-robotics-cover.png"
              alt="Physical AI & Humanoid Robotics Cover"
              className={styles.coverImage}
              style={{
                width: '100%',
                maxWidth: '400px',
                margin: '0 auto',
                display: 'block',
                borderRadius: '8px',
                boxShadow: '0 10px 30px rgba(0,0,0,0.2)'
              }}
            />
          </div>

          {/* Right side: Title and buttons */}
          <div className="col col--6">
            <Heading as="h1" className="hero__title fade-in-up" style={{textAlign: 'left'}}>
              {siteConfig.title}
            </Heading>
            <p className="hero__subtitle fade-in-up" style={{animationDelay: '0.2s', textAlign: 'left', marginTop: '1rem'}}>
              {siteConfig.tagline}
            </p>
            <div className={`${styles.buttons} fade-in-up`} style={{animationDelay: '0.4s', marginTop: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'flex-start'}}>
              <Link
                className="button button--secondary button--lg"
                to="/docs/intro"
                style={{marginBottom: '1rem', width: '200px', textAlign: 'center'}}>
                Start Reading 📚
              </Link>
              <Link
                className="button button--primary button--lg"
                to="/docs/part1/chapter1"
                style={{width: '200px', textAlign: 'center'}}>
                Explore Modules
              </Link>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics`}
      description="Complete educational resource for Physical AI and Humanoid Robotics. AI Systems in the Physical World. Embodied Intelligence.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
      <ChatbotPopup />
    </Layout>
  );
}
