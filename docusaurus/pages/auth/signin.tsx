import React, { useState } from 'react';
import Head from '@docusaurus/Head';
import Layout from '@theme/Layout';
import { useRouter } from 'next/router';
import Link from 'next/link';

export default function SignIn() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // Better Auth integration would go here
      const response = await fetch('/api/auth/signin', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      if (response.ok) {
        const data = await response.json();
        // Store session/token
        localStorage.setItem('sessionToken', data.session_token);
        router.push('/dashboard');
      } else {
        const errorData = await response.json();
        setError(errorData.message || 'Sign in failed');
      }
    } catch (err) {
      setError('An error occurred during sign in');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout title="Sign In" description="Sign in to access the Physical AI & Humanoid Robotics book">
      <Head>
        <title>Sign In | Physical AI & Humanoid Robotics</title>
      </Head>

      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--6 col--offset-3">
            <div className="card">
              <div className="card__header">
                <h2>Sign In</h2>
              </div>

              <div className="card__body">
                {error && (
                  <div className="alert alert--danger" role="alert">
                    {error}
                  </div>
                )}

                <form onSubmit={handleSubmit}>
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

                  <div className="form-group margin-bottom--lg">
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

                  <button
                    type="submit"
                    className="button button--primary button--block"
                    disabled={loading}
                  >
                    {loading ? 'Signing in...' : 'Sign In'}
                  </button>
                </form>
              </div>

              <div className="card__footer">
                <p>
                  Don't have an account?{' '}
                  <Link href="/auth/signup" className="link--secondary">
                    Sign up here
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