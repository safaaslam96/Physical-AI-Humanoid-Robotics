import React from 'react';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';

export default function BlogIndex() {
  const {siteConfig} = useDocusaurusContext();

  const blogPosts = [
    {
      title: "Mastering Sim-to-Real Transfer in Humanoid Robotics",
      description: "Bridging the Reality Gap with Advanced Domain Randomization and System Identification",
      date: "December 2025",
      author: "Physical AI Team",
      readTime: "8 min read",
      slug: "/blog/sim_to_real_transfer_blog",
      category: "Simulation",
      color: "#6f42c1"
    },
    {
      title: "Project Completion: Physical AI & Humanoid Robotics",
      description: "A Comprehensive Educational Resource for Embodied Artificial Intelligence",
      date: "December 2025",
      author: "Physical AI Team",
      readTime: "10 min read",
      slug: "/blog/project_completion_summary",
      category: "Announcement",
      color: "#28a745"
    }
  ];

  return (
    <Layout
      title={`${siteConfig.title} - Blog`}
      description="Latest insights and updates on Physical AI and Humanoid Robotics">
      <div style={{
        padding: '4rem 0',
        maxWidth: '1200px',
        margin: '0 auto',
        textAlign: 'center'
      }}>
        <div style={{
          background: 'linear-gradient(135deg, #0a1f44, #1b3b6f, #6f42c1)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          fontSize: '3rem',
          fontWeight: 'bold',
          marginBottom: '1rem',
          animation: 'glow 2s ease-in-out infinite alternate'
        }}>
          🤖 Physical AI & Humanoid Robotics Blog
        </div>

        <div style={{
          fontSize: '1.4rem',
          color: '#666',
          marginBottom: '3rem',
          textAlign: 'center',
          maxWidth: '800px',
          margin: '0 auto 3rem'
        }}>
          Insights, updates, and deep dives into the world of embodied artificial intelligence and humanoid robotics
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
          gap: '2rem',
          margin: '0 auto'
        }}>
          {blogPosts.map((post, index) => (
            <Link
              key={index}
              to={post.slug}
              style={{
                textDecoration: 'none',
                display: 'block',
                height: '100%'
              }}
            >
              <div
                style={{
                  background: 'rgba(255, 255, 255, 0.08)',
                  backdropFilter: 'blur(16px)',
                  border: '1px solid rgba(255, 255, 255, 0.15)',
                  borderRadius: '20px',
                  padding: '2.5rem',
                  transition: 'all 0.4s cubic-bezier(0.23, 1, 0.32, 1)',
                  overflow: 'hidden',
                  position: 'relative',
                  animation: `float 6s ease-in-out infinite ${index * 0.5}s`,
                  cursor: 'pointer'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-12px) scale(1.03)';
                  e.currentTarget.style.boxShadow = '0 25px 50px rgba(0, 0, 0, 0.25)';
                  e.currentTarget.style.borderColor = `${post.color}66`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0) scale(1)';
                  e.currentTarget.style.boxShadow = 'var(--glass-shadow, 0 8px 32px rgba(31, 38, 135, 0.25))';
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.15)';
                }}
              >
                <div style={{
                  position: 'absolute',
                  top: '-2px',
                  left: '-2px',
                  right: '-2px',
                  bottom: '-2px',
                  background: `linear-gradient(45deg, ${post.color}, ${post.color}80)`,
                  zIndex: -1,
                  borderRadius: '22px',
                  opacity: 0.2,
                  transition: 'opacity 0.3s ease'
                }}></div>

                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '1rem'
                }}>
                  <span style={{
                    backgroundColor: `${post.color}20`,
                    color: post.color,
                    padding: '0.4rem 1rem',
                    borderRadius: '20px',
                    fontSize: '0.9rem',
                    fontWeight: '600',
                    border: `1px solid ${post.color}30`
                  }}>
                    {post.category}
                  </span>
                  <span style={{
                    color: '#888',
                    fontSize: '0.9rem'
                  }}>
                    {post.readTime}
                  </span>
                </div>

                <h2 style={{
                  color: 'white',
                  fontSize: '1.6rem',
                  fontWeight: '700',
                  marginBottom: '1rem',
                  textAlign: 'left',
                  lineHeight: '1.4',
                  background: `linear-gradient(135deg, #ffffff, #e2e8f0)`,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text'
                }}>
                  {post.title}
                </h2>

                <p style={{
                  color: 'rgba(255, 255, 255, 0.85)',
                  textAlign: 'left',
                  lineHeight: '1.6',
                  marginBottom: '1.5rem',
                  fontSize: '1.1rem'
                }}>
                  {post.description}
                </p>

                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginTop: '1.5rem',
                  paddingTop: '1.5rem',
                  borderTop: '1px solid rgba(255, 255, 255, 0.1)'
                }}>
                  <div style={{
                    color: 'rgba(255, 255, 255, 0.7)',
                    fontSize: '0.9rem'
                  }}>
                    <div>{post.date}</div>
                    <div style={{marginTop: '0.2rem'}}>{post.author}</div>
                  </div>
                  <div style={{
                    color: post.color,
                    fontSize: '1.2rem',
                    fontWeight: 'bold'
                  }}>
                    Read More →
                  </div>
                </div>
              </div>
            </Link>
          ))}
        </div>

        <div style={{
          marginTop: '4rem',
          padding: '3rem',
          background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.4), rgba(27, 59, 111, 0.4))',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(255, 255, 255, 0.15)',
          borderRadius: '20px',
          textAlign: 'center'
        }}>
          <h2 style={{
            color: 'white',
            fontSize: '2rem',
            marginBottom: '1rem',
            background: 'linear-gradient(135deg, #ffffff, #e2e8f0)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text'
          }}>
            Stay Updated with Our Latest Research
          </h2>
          <p style={{
            color: 'rgba(255, 255, 255, 0.85)',
            fontSize: '1.2rem',
            marginBottom: '2rem',
            maxWidth: '600px',
            margin: '0 auto 2rem'
          }}>
            Subscribe to our newsletter to receive updates on new blog posts, research findings, and project developments.
          </p>
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '1rem',
            flexWrap: 'wrap'
          }}>
            <input
              type="email"
              placeholder="Enter your email"
              style={{
                padding: '1rem 1.5rem',
                borderRadius: '12px',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                background: 'rgba(255, 255, 255, 0.05)',
                color: 'white',
                fontSize: '1rem',
                minWidth: '300px',
                backdropFilter: 'blur(8px)'
              }}
            />
            <button
              style={{
                backgroundColor: 'linear-gradient(135deg, #6f42c1, #8b5cf6)',
                color: 'white',
                border: 'none',
                padding: '1rem 2rem',
                borderRadius: '12px',
                fontSize: '1rem',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                background: 'linear-gradient(135deg, #6f42c1, #8b5cf6)'
              }}
              onMouseEnter={(e) => {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = '0 10px 30px rgba(111, 66, 193, 0.4)';
              }}
              onMouseLeave={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = '0 8px 25px rgba(111, 66, 193, 0.3)';
              }}
            >
              Subscribe
            </button>
          </div>
        </div>

        <div style={{
          marginTop: '3rem',
          textAlign: 'center',
          color: 'rgba(255, 255, 255, 0.6)',
          fontSize: '0.9rem'
        }}>
          <p>More blog posts coming soon as we continue to advance the field of Physical AI and Humanoid Robotics.</p>
        </div>
      </div>

      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
        }

        @keyframes glow {
          from { text-shadow: 0 0 5px rgba(255, 255, 255, 0.2); }
          to { text-shadow: 0 0 20px #6f42c1, 0 0 30px rgba(111, 66, 193, 0.3); }
        }
      `}</style>
    </Layout>
  );
}