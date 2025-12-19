import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

export default function AllContent() {
  return (
    <Layout
      title="Complete Book Content - Physical AI & Humanoid Robotics"
      description="All chapters and content from the Physical AI & Humanoid Robotics book">
      <div style={{
        padding: '4rem 0',
        maxWidth: '1200px',
        margin: '0 auto'
      }}>
        <div style={{
          textAlign: 'center',
          marginBottom: '3rem'
        }}>
          <h1 style={{
            background: 'linear-gradient(135deg, #0a1f44, #1b3b6f, #6f42c1)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            fontSize: '3rem',
            fontWeight: 'bold',
            marginBottom: '1rem'
          }}>
            Complete Book Content
          </h1>
          <p style={{
            fontSize: '1.2rem',
            color: '#666',
            maxWidth: '800px',
            margin: '0 auto'
          }}>
            Explore all chapters and resources from the comprehensive Physical AI & Humanoid Robotics book
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '2rem',
          marginBottom: '3rem'
        }}>
          {/* Part 1: Introduction */}
          <div style={{
            backgroundColor: 'white',
            padding: '2rem',
            borderRadius: '12px',
            border: '2px solid #007cba',
            boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
            transition: 'transform 0.3s ease, box-shadow 0.3s ease'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-5px)';
            e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
          }}>
            <h3 style={{color: '#0a1f44', fontSize: '1.5rem', marginBottom: '1rem'}}>Part 1: Introduction to Physical AI</h3>
            <ul style={{textAlign: 'left', marginBottom: '1rem'}}>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part1/chapter1">Chapter 1: Foundations of Physical AI</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part1/chapter2">Chapter 2: Humanoid Robotics Landscape</Link></li>
            </ul>
            <Link
              to="/Physical-AI-Humanoid-Robotics/docs/part1/chapter1"
              style={{
                display: 'inline-block',
                backgroundColor: '#007cba',
                color: 'white',
                padding: '0.5rem 1rem',
                textDecoration: 'none',
                borderRadius: '6px',
                fontSize: '0.9rem'
              }}
            >
              Start Reading →
            </Link>
          </div>

          {/* Part 2: ROS 2 */}
          <div style={{
            backgroundColor: 'white',
            padding: '2rem',
            borderRadius: '12px',
            border: '2px solid #28a745',
            boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
            transition: 'transform 0.3s ease, box-shadow 0.3s ease'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-5px)';
            e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
          }}>
            <h3 style={{color: '#0a1f44', fontSize: '1.5rem', marginBottom: '1rem'}}>Part 2: Robotic Nervous System (ROS 2)</h3>
            <ul style={{textAlign: 'left', marginBottom: '1rem'}}>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part2/chapter3">Chapter 3: ROS 2 Architecture</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part2/chapter4">Chapter 4: Building ROS 2 Packages</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part2/chapter5">Chapter 5: ROS 2 Communication Patterns</Link></li>
            </ul>
            <Link
              to="/Physical-AI-Humanoid-Robotics/docs/part2/chapter3"
              style={{
                display: 'inline-block',
                backgroundColor: '#28a745',
                color: 'white',
                padding: '0.5rem 1rem',
                textDecoration: 'none',
                borderRadius: '6px',
                fontSize: '0.9rem'
              }}
            >
              Start Reading →
            </Link>
          </div>

          {/* Part 3: Simulation */}
          <div style={{
            backgroundColor: 'white',
            padding: '2rem',
            borderRadius: '12px',
            border: '2px solid #ffc107',
            boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
            transition: 'transform 0.3s ease, box-shadow 0.3s ease'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-5px)';
            e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
          }}>
            <h3 style={{color: '#0a1f44', fontSize: '1.5rem', marginBottom: '1rem'}}>Part 3: Digital Twin (Gazebo & Unity)</h3>
            <ul style={{textAlign: 'left', marginBottom: '1rem'}}>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part3/chapter6">Chapter 6: Gazebo Simulation</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part3/chapter7">Chapter 7: URDF/SDF Formats</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part3/chapter8">Chapter 8: Unity Visualization</Link></li>
            </ul>
            <Link
              to="/Physical-AI-Humanoid-Robotics/docs/part3/chapter6"
              style={{
                display: 'inline-block',
                backgroundColor: '#ffc107',
                color: '#0a1f44',
                padding: '0.5rem 1rem',
                textDecoration: 'none',
                borderRadius: '6px',
                fontSize: '0.9rem',
                fontWeight: 'bold'
              }}
            >
              Start Reading →
            </Link>
          </div>

          {/* Part 4: AI-Brain */}
          <div style={{
            backgroundColor: 'white',
            padding: '2rem',
            borderRadius: '12px',
            border: '2px solid #6f42c1',
            boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
            transition: 'transform 0.3s ease, box-shadow 0.3s ease'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-5px)';
            e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
          }}>
            <h3 style={{color: '#0a1f44', fontSize: '1.5rem', marginBottom: '1rem'}}>Part 4: AI-Robot Brain (NVIDIA Isaac™)</h3>
            <ul style={{textAlign: 'left', marginBottom: '1rem'}}>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part4/chapter9">Chapter 9: Isaac SDK & Isaac Sim</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part4/chapter10">Chapter 10: Isaac ROS</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part4/chapter11">Chapter 11: Nav2 for Humanoids</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part4/chapter12">Chapter 12: Sim-to-Real Transfer</Link></li>
            </ul>
            <Link
              to="/Physical-AI-Humanoid-Robotics/docs/part4/chapter9"
              style={{
                display: 'inline-block',
                backgroundColor: '#6f42c1',
                color: 'white',
                padding: '0.5rem 1rem',
                textDecoration: 'none',
                borderRadius: '6px',
                fontSize: '0.9rem'
              }}
            >
              Start Reading →
            </Link>
          </div>

          {/* Part 5: Humanoid Development */}
          <div style={{
            backgroundColor: 'white',
            padding: '2rem',
            borderRadius: '12px',
            border: '2px solid #17a2b8',
            boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
            transition: 'transform 0.3s ease, box-shadow 0.3s ease'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-5px)';
            e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
          }}>
            <h3 style={{color: '#0a1f44', fontSize: '1.5rem', marginBottom: '1rem'}}>Part 5: Humanoid Robot Development</h3>
            <ul style={{textAlign: 'left', marginBottom: '1rem'}}>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part5/chapter13">Chapter 13: Kinematics & Dynamics</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part5/chapter14">Chapter 14: Bipedal Locomotion</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part5/chapter15">Chapter 15: Manipulation & Grasping</Link></li>
            </ul>
            <Link
              to="/Physical-AI-Humanoid-Robotics/docs/part5/chapter13"
              style={{
                display: 'inline-block',
                backgroundColor: '#17a2b8',
                color: 'white',
                padding: '0.5rem 1rem',
                textDecoration: 'none',
                borderRadius: '6px',
                fontSize: '0.9rem'
              }}
            >
              Start Reading →
            </Link>
          </div>

          {/* Part 6: Vision-Language-Action */}
          <div style={{
            backgroundColor: 'white',
            padding: '2rem',
            borderRadius: '12px',
            border: '2px solid #fd7e14',
            boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
            transition: 'transform 0.3s ease, box-shadow 0.3s ease'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-5px)';
            e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
          }}>
            <h3 style={{color: '#0a1f44', fontSize: '1.5rem', marginBottom: '1rem'}}>Part 6: Vision-Language-Action & Capstone</h3>
            <ul style={{textAlign: 'left', marginBottom: '1rem'}}>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part6/chapter16">Chapter 16: Natural Human-Robot Interaction</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part6/chapter17">Chapter 17: LLM Integration</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part6/chapter18">Chapter 18: Speech Recognition</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part6/chapter19">Chapter 19: Cognitive Planning</Link></li>
              <li><Link to="/Physical-AI-Humanoid-Robotics/docs/part6/chapter20">Chapter 20: Autonomous Humanoid Capstone</Link></li>
            </ul>
            <Link
              to="/Physical-AI-Humanoid-Robotics/docs/part6/chapter16"
              style={{
                display: 'inline-block',
                backgroundColor: '#fd7e14',
                color: 'white',
                padding: '0.5rem 1rem',
                textDecoration: 'none',
                borderRadius: '6px',
                fontSize: '0.9rem'
              }}
            >
              Start Reading →
            </Link>
          </div>
        </div>

        <div style={{
          backgroundColor: '#f8f9fa',
          padding: '3rem',
          borderRadius: '12px',
          marginBottom: '3rem',
          border: '2px solid #dee2e6'
        }}>
          <h2 style={{color: '#0a1f44', fontSize: '2rem', marginBottom: '1.5rem', textAlign: 'center'}}>
            Additional Resources
          </h2>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '1.5rem'
          }}>
            <div style={{
              backgroundColor: 'white',
              padding: '1.5rem',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <h3 style={{color: '#6f42c1', fontSize: '1.3rem', marginBottom: '1rem'}}>AI Chatbot</h3>
              <p>Interactive AI assistant for questions and guidance</p>
              <Link
                to="/Physical-AI-Humanoid-Robotics/chatbot"
                style={{
                  display: 'inline-block',
                  backgroundColor: '#6f42c1',
                  color: 'white',
                  padding: '0.5rem 1rem',
                  textDecoration: 'none',
                  borderRadius: '6px',
                  marginTop: '0.5rem'
                }}
              >
                Try Now
              </Link>
            </div>
            <div style={{
              backgroundColor: 'white',
              padding: '1.5rem',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <h3 style={{color: '#28a745', fontSize: '1.3rem', marginBottom: '1rem'}}>Code Examples</h3>
              <p>Complete implementation examples and tutorials</p>
              <Link
                to="/Physical-AI-Humanoid-Robotics/docs/code-examples"
                style={{
                  display: 'inline-block',
                  backgroundColor: '#28a745',
                  color: 'white',
                  padding: '0.5rem 1rem',
                  textDecoration: 'none',
                  borderRadius: '6px',
                  marginTop: '0.5rem'
                }}
              >
                Browse Code
              </Link>
            </div>
            <div style={{
              backgroundColor: 'white',
              padding: '1.5rem',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <h3 style={{color: '#17a2b8', fontSize: '1.3rem', marginBottom: '1rem'}}>Video Tutorials</h3>
              <p>Step-by-step video guides for complex concepts</p>
              <Link
                to="/Physical-AI-Humanoid-Robotics/videos"
                style={{
                  display: 'inline-block',
                  backgroundColor: '#17a2b8',
                  color: 'white',
                  padding: '0.5rem 1rem',
                  textDecoration: 'none',
                  borderRadius: '6px',
                  marginTop: '0.5rem'
                }}
              >
                Watch Videos
              </Link>
            </div>
            <div style={{
              backgroundColor: 'white',
              padding: '1.5rem',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <h3 style={{color: '#ffc107', fontSize: '1.3rem', marginBottom: '1rem'}}>Community</h3>
              <p>Join our community of Physical AI researchers</p>
              <Link
                to="/Physical-AI-Humanoid-Robotics/community"
                style={{
                  display: 'inline-block',
                  backgroundColor: '#ffc107',
                  color: '#0a1f44',
                  padding: '0.5rem 1rem',
                  textDecoration: 'none',
                  borderRadius: '6px',
                  fontWeight: 'bold',
                  marginTop: '0.5rem'
                }}
              >
                Join Community
              </Link>
            </div>
          </div>
        </div>

        <div style={{
          textAlign: 'center',
          padding: '2rem',
          background: 'linear-gradient(135deg, #0a1f44, #1b3b6f)',
          color: 'white',
          borderRadius: '12px'
        }}>
          <h2 style={{fontSize: '2rem', marginBottom: '1rem'}}>Ready to Start Your Journey?</h2>
          <p style={{fontSize: '1.2rem', marginBottom: '2rem', opacity: 0.9}}>
            Begin with the introduction and work your way through the comprehensive curriculum
          </p>
          <Link
            to="/Physical-AI-Humanoid-Robotics/docs/intro"
            style={{
              backgroundColor: '#6f42c1',
              color: 'white',
              padding: '1rem 2rem',
              textDecoration: 'none',
              borderRadius: '8px',
              fontSize: '1.2rem',
              fontWeight: 'bold',
              transition: 'all 0.3s ease'
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = 'translateY(-3px)';
              e.target.style.boxShadow = '0 8px 25px rgba(111, 66, 193, 0.3)';
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = '0 4px 15px rgba(111, 66, 193, 0.2)';
            }}
          >
            Start Reading the Book →
          </Link>
        </div>
      </div>
    </Layout>
  );
}