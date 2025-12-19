import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

// Book Cover Section Component
function BookCoverSection(): ReactNode {
  return (
    <section className={styles.bookCoverSection}>
      <div className="container">
        <div className={clsx(styles.bookCoverLayout)}>
          <div className={styles.bookCoverImage}>
            <img
              src="/static/img/physical-ai-humanoid-robotics-cover.png"
              alt="Physical AI & Humanoid Robotics Book Cover"
              style={{width: '100%', height: 'auto', borderRadius: '12px'}}
            />
          </div>
          <div className={styles.bookInfo}>
            <h1 className={styles.bookTitle}>Physical AI & Humanoid Robotics</h1>
            <p className={styles.bookDescription}>
              Complete educational resource for AI Systems in the Physical World. Embodied Intelligence.
              This comprehensive book bridges the gap between digital AI and physical robots, providing
              hands-on experience with ROS 2, Gazebo simulation, NVIDIA Isaac SDK, and advanced AI techniques.
            </p>
            <div className={styles.ctaButtons}>
              <Link className={clsx(styles.primaryButton)} to="/docs/intro">
                Start Reading
              </Link>
              <Link className={clsx(styles.secondaryButton)} to="/docs/part1/chapter1">
                Explore Modules
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// Modules Section Component
function ModulesSection(): ReactNode {
  const modules = [
    {
      icon: '🤖',
      title: 'The Robotic Nervous System',
      description: 'Master ROS 2 architecture, nodes, topics, and services. Build packages with Python and bridge AI agents to ROS controllers.',
      link: '/docs/part2/chapter3'
    },
    {
      icon: '🎮',
      title: 'The Digital Twin',
      description: 'Set up Gazebo simulation environments, work with URDF/SDF robot descriptions, and explore Unity visualization.',
      link: '/docs/part2/chapter4'
    },
    {
      icon: '🧠',
      title: 'The AI-Robot Brain',
      description: 'Explore NVIDIA Isaac SDK, Isaac ROS, Nav2 path planning, and sim-to-real transfer techniques.',
      link: '/docs/part2/chapter5'
    },
    {
      icon: '🗣️',
      title: 'Vision-Language-Action',
      description: 'Integrate LLMs for conversational AI, implement speech recognition, and build cognitive planning systems.',
      link: '/docs/part1/chapter2'
    }
  ];

  return (
    <section className={styles.modulesSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>Main Learning Modules</Heading>
        <p className={styles.sectionSubtitle}>Four comprehensive modules covering essential topics in Physical AI and Humanoid Robotics</p>

        <div className={styles.modulesGrid}>
          {modules.map((module, idx) => (
            <div key={idx} className={styles.moduleCard}>
              <div className={styles.moduleIcon}>{module.icon}</div>
              <h3 className={styles.moduleTitle}>{module.title}</h3>
              <p className={styles.moduleDescription}>{module.description}</p>
              <Link className="button button--outline button--primary button--sm" to={module.link}>
                Start Learning
              </Link>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// Book Chapters Section Component
function ChaptersSection(): ReactNode {
  const chapters = [
    {
      part: 'Part I',
      title: 'Introduction to Physical AI',
      description: 'Foundations of Physical AI and Embodied Intelligence'
    },
    {
      part: 'Part II',
      title: 'The Robotic Nervous System',
      description: 'ROS 2 Architecture and Core Concepts'
    },
    {
      part: 'Part III',
      title: 'The Digital Twin',
      description: 'Gazebo Simulation and Robot Description'
    },
    {
      part: 'Part IV',
      title: 'The AI-Robot Brain',
      description: 'NVIDIA Isaac SDK and Path Planning'
    },
    {
      part: 'Part V',
      title: 'Humanoid Robot Development',
      description: 'Kinematics, Locomotion, and Manipulation'
    },
    {
      part: 'Part VI',
      title: 'Vision-Language-Action & Capstone',
      description: 'LLMs Integration and Autonomous Systems'
    }
  ];

  return (
    <section className={styles.chaptersSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>Book Structure</Heading>
        <p className={styles.sectionSubtitle}>Progressive learning path from fundamentals to advanced humanoid robotics</p>

        <div className={styles.chaptersGrid}>
          {chapters.map((chapter, idx) => (
            <div key={idx} className={styles.chapterCard}>
              <div className={styles.chapterNumber}>{chapter.part}</div>
              <h3 className={styles.chapterTitle}>{chapter.title}</h3>
              <p className={styles.chapterDescription}>{chapter.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <>
      <BookCoverSection />
      <ModulesSection />
      <ChaptersSection />
    </>
  );
}
