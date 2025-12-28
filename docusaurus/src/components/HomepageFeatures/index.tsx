import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';
import { motion } from 'framer-motion';
import { Swiper, SwiperSlide } from 'swiper/react';
import { Pagination, Autoplay } from 'swiper/modules';
import 'swiper/css';
import 'swiper/css/pagination';
import 'swiper/css/autoplay';

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
    }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: "easeOut"
    }
  }
};

// Book Cover Section Component
function BookCoverSection(): ReactNode {
  return (
    <section className={styles.bookCoverSection}>
      <div className="container">
        <motion.div
          className={styles.bookCoverLayout}
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
        >
          <motion.div
            className={styles.bookCoverImage}
            variants={itemVariants}
          >
            <img
              src="/img/physical-ai-humanoid-robotics-cover.png"
              alt="Physical AI & Humanoid Robotics Book Cover"
            />
          </motion.div>
          <motion.div
            className={styles.bookInfo}
            variants={itemVariants}
          >
            <motion.div
              className={styles.bookTitle}
              variants={itemVariants}
            >
              <h1>Physical AI & Humanoid Robotics</h1>
            </motion.div>
            <motion.div
              className={styles.bookDescription}
              variants={itemVariants}
            >
              <p>
                Complete educational resource for AI Systems in the Physical World. Embodied Intelligence.
                This comprehensive book bridges the gap between digital AI and physical robots, providing
                hands-on experience with ROS 2, Gazebo simulation, NVIDIA Isaac SDK, and advanced AI techniques.
              </p>
            </motion.div>
            <motion.div
              className={styles.ctaButtons}
              variants={itemVariants}
            >
              <Link
                className="button button--primary button--lg"
                to="/docs/intro"
              >
                Start Reading
              </Link>
              <Link
                className="button button--secondary button--lg"
                to="/docs/part1/chapter1"
              >
                Explore Modules
              </Link>
            </motion.div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}

// Quarter Overview Section Component
function QuarterOverviewSection(): ReactNode {
  return (
    <section className={styles.quarterOverviewSection}>
      <div className="container">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <Heading as="h2" className={styles.sectionTitle}>Quarter Overview</Heading>
          <p className={styles.sectionSubtitle}>A comprehensive capstone experience in Physical AI and Humanoid Robotics</p>

          <div className={styles.quarterOverviewContent}>
            <p>
              The future of AI extends beyond digital spaces into the physical world. This capstone quarter introduces Physical AI—AI systems that function in reality and comprehend physical laws. Students learn to design, simulate, and deploy humanoid robots capable of natural human interactions using ROS 2, Gazebo, and NVIDIA Isaac.
            </p>

            <div className={styles.learningOutcomes}>
              <h3>Learning Outcomes:</h3>
              <div className={styles.learningOutcomesGrid}>
                <motion.div
                  className={styles.learningOutcomeCard}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: 0.1 }}
                  whileHover={{ y: -5, scale: 1.02, transition: { duration: 0.3 } }}
                >
                  <div className={styles.learningOutcomeIcon}>🧠</div>
                  <h4 className={styles.learningOutcomeTitle}>Understand Physical AI principles and embodied intelligence</h4>
                  <p className={styles.learningOutcomeDescription}>Learn how AI systems function in the physical world and comprehend physical laws</p>
                  <Link className={`button button--primary button--sm ${styles.learningOutcomeButton}`} to="/docs/part1/chapter1">
                    Learn More
                  </Link>
                </motion.div>

                <motion.div
                  className={styles.learningOutcomeCard}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: 0.2 }}
                  whileHover={{ y: -5, scale: 1.02, transition: { duration: 0.3 } }}
                >
                  <div className={styles.learningOutcomeIcon}>🤖</div>
                  <h4 className={styles.learningOutcomeTitle}>Master ROS 2 (Robot Operating System) for robotic control</h4>
                  <p className={styles.learningOutcomeDescription}>Learn ROS 2 Nodes, Topics, Services and how to bridge Python agents to ROS controllers</p>
                  <Link className={`button button--primary button--sm ${styles.learningOutcomeButton}`} to="/docs/part1/chapter1">
                    Learn More
                  </Link>
                </motion.div>

                <motion.div
                  className={styles.learningOutcomeCard}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: 0.3 }}
                  whileHover={{ y: -5, scale: 1.02, transition: { duration: 0.3 } }}
                >
                  <div className={styles.learningOutcomeIcon}>🎮</div>
                  <h4 className={styles.learningOutcomeTitle}>Simulate robots with Gazebo and Unity</h4>
                  <p className={styles.learningOutcomeDescription}>Build physics simulations, understand gravity and collisions, and create human-robot interactions</p>
                  <Link className={`button button--primary button--sm ${styles.learningOutcomeButton}`} to="/docs/part2/chapter3">
                    Learn More
                  </Link>
                </motion.div>

                <motion.div
                  className={styles.learningOutcomeCard}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: 0.4 }}
                  whileHover={{ y: -5, scale: 1.02, transition: { duration: 0.3 } }}
                >
                  <div className={styles.learningOutcomeIcon}>🦾</div>
                  <h4 className={styles.learningOutcomeTitle}>Develop with NVIDIA Isaac AI robot platform</h4>
                  <p className={styles.learningOutcomeDescription}>Use Isaac Sim for photorealistic simulation and Isaac ROS for hardware-accelerated navigation</p>
                  <Link className={`button button--primary button--sm ${styles.learningOutcomeButton}`} to="/docs/part2/chapter4">
                    Learn More
                  </Link>
                </motion.div>

                <motion.div
                  className={styles.learningOutcomeCard}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: 0.5 }}
                  whileHover={{ y: -5, scale: 1.02, transition: { duration: 0.3 } }}
                >
                  <div className={styles.learningOutcomeIcon}>🦾</div>
                  <h4 className={styles.learningOutcomeTitle}>Design humanoid robots for natural interactions</h4>
                  <p className={styles.learningOutcomeDescription}>Learn humanoid robot kinematics, bipedal locomotion, and natural interaction design</p>
                  <Link className={`button button--primary button--sm ${styles.learningOutcomeButton}`} to="/docs/part2/chapter5">
                    Learn More
                  </Link>
                </motion.div>

                <motion.div
                  className={styles.learningOutcomeCard}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: 0.6 }}
                  whileHover={{ y: -5, scale: 1.02, transition: { duration: 0.3 } }}
                >
                  <div className={styles.learningOutcomeIcon}>🗣️</div>
                  <h4 className={styles.learningOutcomeTitle}>Integrate GPT models for conversational robotics</h4>
                  <p className={styles.learningOutcomeDescription}>Combine LLMs with robotics for voice commands and natural language processing</p>
                  <Link className={`button button--primary button--sm ${styles.learningOutcomeButton}`} to="/docs/part2/chapter5">
                    Learn More
                  </Link>
                </motion.div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

// Modules Carousel Section Component (PERFECT & RESPONSIVE)
function ModulesSection(): ReactNode {
  const modules = [
    {
      icon: '🤖',
      title: 'Module 1: The Robotic Nervous System',
      description: 'Focus: Middleware for robot control. ROS 2 Nodes, Topics, and Services. Bridging Python Agents to ROS controllers using rclpy. Understanding URDF (Unified Robot Description Format) for humanoids.',
      link: '/docs/part1/chapter1'
    },
    {
      icon: '🎮',
      title: 'Module 2: The Digital Twin',
      description: 'Focus: Physics simulation and environment building. Simulating physics, gravity, and collisions in Gazebo. High-fidelity rendering and human-robot interaction in Unity. Simulating sensors: LiDAR, Depth Cameras, and IMUs.',
      link: '/docs/part2/chapter3'
    },
    {
      icon: '🧠',
      title: 'Module 3: The AI-Robot Brain',
      description: 'Focus: Advanced perception and training. NVIDIA Isaac Sim: Photorealistic simulation and synthetic data generation. Isaac ROS: Hardware-accelerated VSLAM (Visual SLAM) and navigation. Nav2: Path planning for bipedal humanoid movement.',
      link: '/docs/part2/chapter4'
    },
    {
      icon: '🗣️',
      title: 'Module 4: Vision-Language-Action',
      description: 'Focus: The convergence of LLMs and Robotics. Voice-to-Action: Using OpenAI Whisper for voice commands. Cognitive Planning: Using LLMs to translate natural language ("Clean the room") into a sequence of ROS 2 actions. Capstone Project: The Autonomous Humanoid.',
      link: '/docs/part2/chapter5'
    }
  ];

  // Group modules into 2 pages (first half and second half)
  const groupedModules = [
    modules.slice(0, 2), // First page: Module 1 and 2
    modules.slice(2, 4)  // Second page: Module 3 and 4
  ];

  return (
    <section className={styles.modulesSection}>
      <div className="container">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <Heading as="h2" className={styles.sectionTitle}>Main Learning Modules</Heading>
          <p className={styles.sectionSubtitle}>Four comprehensive modules covering essential topics in Physical AI and Humanoid Robotics</p>

          <div className={styles.modulesCarousel}>
            <Swiper
              modules={[Pagination, Autoplay]}
              spaceBetween={30}
              slidesPerView={1}
              pagination={{
                clickable: true,
                type: 'bullets'
              }}
              autoplay={{
                delay: 5000,
                disableOnInteraction: false,
                pauseOnMouseEnter: true,
              }}
              loop={false}
              grabCursor={true}
              breakpoints={{
                640: { slidesPerView: 1, spaceBetween: 20 },
                768: { slidesPerView: 1, spaceBetween: 30 }, // Show 1 page at a time on medium screens
                1024: { slidesPerView: 1, spaceBetween: 40 }, // Show 1 page at a time on desktop
              }}
              navigation={false}
            >
              {groupedModules.map((group, groupIndex) => (
                <SwiperSlide key={groupIndex}>
                  <div className={styles.modulesGroup}>
                    {group.map((module, idx) => (
                      <motion.div
                        className={styles.moduleCard}
                        initial={{ opacity: 0, scale: 0.9 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.4 }}
                        whileHover={{ y: -5, scale: 1.02, transition: { duration: 0.3 } }}
                        key={idx}
                      >
                        <div className={styles.moduleIcon}>{module.icon}</div>
                        <h3 className={styles.moduleTitle}>{module.title}</h3>
                        <p className={styles.moduleDescription}>{module.description}</p>
                        <Link className="button button--primary button--sm" to={module.link}>
                          Explore Module
                        </Link>
                      </motion.div>
                    ))}
                  </div>
                </SwiperSlide>
              ))}
            </Swiper>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

// Why Physical AI Matters Section (Keep your existing one if it's good, or use this placeholder)
function WhyPhysicalAIMattersSection(): ReactNode {
  const reasonsCards = [
    {
      icon: '🧠',
      title: 'Bridging Digital and Physical AI',
      description: 'Move beyond screen-based AI to systems that understand and interact with the physical world, enabling robots to navigate, manipulate, and respond to real-world physics.'
    },
    {
      icon: '🤖',
      title: 'Embodied Intelligence',
      description: 'Learn how AI systems function when they have physical form and must comprehend physical laws, gravity, and real-world constraints.'
    },
    {
      icon: '🎯',
      title: 'Real-World Applications',
      description: 'From warehouse automation to personal assistants, humanoid robots are the next frontier in AI deployment with tangible societal impact.'
    }
  ];

  const weeklyCards = [
    {
      week: 'Weeks 1-2',
      title: 'Introduction to Physical AI',
      topics: 'Foundations of Physical AI and embodied intelligence. From digital AI to robots that understand physical laws. Overview of humanoid robotics landscape. Sensor systems: LIDAR, cameras, IMUs, force/torque sensors.'
    },
    {
      week: 'Weeks 3-5',
      title: 'ROS 2 Fundamentals',
      topics: 'ROS 2 architecture and core concepts. Nodes, topics, services, and actions. Building ROS 2 packages with Python. Launch files and parameter management.'
    },
    {
      week: 'Weeks 6-7',
      title: 'Robot Simulation with Gazebo',
      topics: 'Gazebo simulation environment setup. URDF and SDF robot description formats. Physics simulation and sensor simulation. Introduction to Unity for robot visualization.'
    },
    {
      week: 'Weeks 8-10',
      title: 'NVIDIA Isaac Platform',
      topics: 'NVIDIA Isaac SDK and Isaac Sim. AI-powered perception and manipulation. Reinforcement learning for robot control. Sim-to-real transfer techniques.'
    },
    {
      week: 'Weeks 11-12',
      title: 'Humanoid Robot Development',
      topics: 'Humanoid robot kinematics and dynamics. Bipedal locomotion and balance control. Manipulation and grasping with humanoid hands. Natural human-robot interaction design.'
    },
    {
      week: 'Week 13',
      title: 'Conversational Robotics',
      topics: 'Integrating GPT models for conversational AI in robots. Speech recognition and natural language understanding. Multi-modal interaction: speech, gesture, vision.'
    }
  ];

  return (
    <section className={styles.whyPhysicalAISection}>
      <div className="container">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <Heading as="h2" className={styles.sectionTitle}>Why Physical AI Matters</Heading>
          <p className={styles.sectionSubtitle}>Understanding the importance of AI systems that function in the physical world</p>

          {/* Reasons Cards Grid */}
          <div className={styles.reasonsGrid}>
            {reasonsCards.map((reason, index) => (
              <motion.div
                key={index}
                className={styles.reasonCard}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                whileHover={{ y: -5, scale: 1.02, transition: { duration: 0.3 } }}
              >
                <div className={styles.reasonIcon}>{reason.icon}</div>
                <h3 className={styles.reasonTitle}>{reason.title}</h3>
                <p className={styles.reasonDescription}>{reason.description}</p>
              </motion.div>
            ))}
          </div>

          <Heading as="h3" className={styles.weeklyBreakdownTitle}>Weekly Breakdown</Heading>
          <div className={styles.weeklyGrid}>
            {weeklyCards.map((card, index) => (
              <Link
                key={index}
                to={
                  index === 0 ? "/docs/intro" :
                  index === 1 ? "/docs/part1/chapter1" :
                  index === 2 ? "/docs/part2/chapter3" :
                  index === 3 ? "/docs/part2/chapter4" :
                  index === 4 ? "/docs/part2/chapter5" :
                  "/docs/part2/chapter5"
                }
                className={styles.weekCardLink}
              >
                <motion.div
                  className={styles.weekCard}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-100px" }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                >
                  <div className={styles.weekHeader}>
                    <div className={styles.weekNumber}>{card.week}</div>
                    <h3 className={styles.weekTitle}>{card.title}</h3>
                  </div>
                  <p className={styles.weekTopics}>{card.topics}</p>
                </motion.div>
              </Link>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <>
      <BookCoverSection />
      <QuarterOverviewSection />
      <ModulesSection />
      <WhyPhysicalAIMattersSection />
    </>
  );
}