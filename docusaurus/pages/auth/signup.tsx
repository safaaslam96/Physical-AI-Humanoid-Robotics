import React, { useState } from 'react';
import Head from '@docusaurus/Head';
import Layout from '@theme/Layout';
import { useRouter } from 'next/router';
import Link from 'next/link';

export default function SignUp() {
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [expertiseLevel, setExpertiseLevel] = useState('beginner');
  const [domain, setDomain] = useState('software');
  const [interests, setInterests] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // Better Auth integration would go here
      const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
          name,
          password,
          background: {
            domain,
            expertise_level: expertiseLevel,
            interests,
          },
        }),
      });

      if (response.ok) {
        const data = await response.json();
        // Store session/token
        localStorage.setItem('sessionToken', data.session_token);
        router.push('/dashboard');
      } else {
        const errorData = await response.json();
        setError(errorData.message || 'Sign up failed');
      }
    } catch (err) {
      setError('An error occurred during sign up');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleInterestChange = (interest: string) => {
    if (interests.includes(interest)) {
      setInterests(interests.filter(i => i !== interest));
    } else {
      setInterests([...interests, interest]);
    }
  };

  return (
    <Layout title="Sign Up" description="Sign up to access personalized content in Physical AI & Humanoid Robotics book">
      <Head>
        <title>Sign Up | Physical AI & Humanoid Robotics</title>
      </Head>

      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--6 col--offset-3">
            <div className="card">
              <div className="card__header">
                <h2>Sign Up</h2>
              </div>

              <div className="card__body">
                {error && (
                  <div className="alert alert--danger" role="alert">
                    {error}
                  </div>
                )}

                <form onSubmit={handleSubmit}>
                  <div className="form-group margin-bottom--md">
                    <label htmlFor="name">Full Name</label>
                    <input
                      type="text"
                      id="name"
                      className="form-control"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      required
                    />
                  </div>

                  <div className="form-group margin-bottom--md">
                    <label htmlFor="email">Email</label>
                    <input
                      type="email"
                      id="email"
                      className="form-control"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  </div>

                  <div className="form-group margin-bottom--md">
                    <label htmlFor="password">Password</label>
                    <input
                      type="password"
                      id="password"
                      className="form-control"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                    />
                  </div>

                  <div className="form-group margin-bottom--md">
                    <label>Expertise Level</label>
                    <div className="form-group">
                      {['beginner', 'intermediate', 'advanced'].map((level) => (
                        <label key={level} className="radio">
                          <input
                            type="radio"
                            name="expertiseLevel"
                            checked={expertiseLevel === level}
                            onChange={() => setExpertiseLevel(level)}
                          />
                          <span className="margin-left--sm">
                            {level.charAt(0).toUpperCase() + level.slice(1)}
                          </span>
                        </label>
                      ))}
                    </div>
                  </div>

                  <div className="form-group margin-bottom--md">
                    <label>Domain Background</label>
                    <select
                      className="form-control"
                      value={domain}
                      onChange={(e) => setDomain(e.target.value)}
                    >
                      <option value="software">Software</option>
                      <option value="hardware">Hardware</option>
                      <option value="both">Both</option>
                      <option value="other">Other</option>
                    </select>
                  </div>

                  <div className="form-group margin-bottom--md">
                    <label>Interests (Select all that apply)</label>
                    <div className="form-group">
                      {['Robotics', 'AI', 'Machine Learning', 'Computer Vision', 'Natural Language Processing', 'Humanoid Design'].map((interest) => (
                        <label key={interest} className="checkbox">
                          <input
                            type="checkbox"
                            checked={interests.includes(interest)}
                            onChange={() => handleInterestChange(interest)}
                          />
                          <span className="margin-left--sm">{interest}</span>
                        </label>
                      ))}
                    </div>
                  </div>

                  <button
                    type="submit"
                    className="button button--primary button--block"
                    disabled={loading}
                  >
                    {loading ? 'Creating Account...' : 'Sign Up'}
                  </button>
                </form>
              </div>

              <div className="card__footer">
                <p>
                  Already have an account?{' '}
                  <Link href="/auth/signin" className="link--secondary">
                    Sign in here
                  </Link>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}