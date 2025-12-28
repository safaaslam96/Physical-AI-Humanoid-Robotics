import React, { useState } from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import { motion } from 'framer-motion';

export default function Signin() {
  const { siteConfig } = useDocusaurusContext();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});

  // Validate individual fields
  const validateField = (name, value) => {
    switch (name) {
      case 'email':
        if (!value) return 'Email is required';
        if (!/\S+@\S+\.\S+/.test(value)) return 'Email address is invalid';
        return '';
      case 'password':
        if (!value) return 'Password is required';
        if (value.length < 6) return 'Password must be at least 6 characters';
        return '';
      default:
        return '';
    }
  };

  // Validate entire form
  const validateForm = () => {
    const newErrors = {};
    newErrors.email = validateField('email', email);
    newErrors.password = validateField('password', password);

    setErrors(newErrors);
    return !Object.values(newErrors).some(error => error !== '');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (validateForm()) {
      setIsLoading(true);
      try {
        // In a real implementation, this would call BetterAuth's sign-in API
        await new Promise(resolve => setTimeout(resolve, 1500));
        console.log('Sign in attempt with:', { email, password });
        alert('Sign in successful! (This is a simulation)');
      } catch (error) {
        console.error('Sign in error:', error);
        alert('Sign in failed. Please try again.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleBlur = (fieldName) => {
    setTouched({ ...touched, [fieldName]: true });
    setErrors({ ...errors, [fieldName]: validateField(fieldName, fieldName === 'email' ? email : password) });
  };

  // Handle Google OAuth sign in
  const handleGoogleSignIn = () => {
    // In a real implementation, this would redirect to BetterAuth's Google OAuth endpoint
    console.log('Initiating Google OAuth sign in');
  };

  return (
    <Layout title={`Sign In - ${siteConfig.title}`} description="Sign in to your Physical AI & Humanoid Robotics account">
      <main className="auth-page">
        <motion.div
          className="auth-container"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="auth-card">
            <div className="auth-header">
              <motion.h1
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                Welcome Back
              </motion.h1>
              <motion.p
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                Sign in to access your account
              </motion.p>
            </div>

            <motion.form
              onSubmit={handleSubmit}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="auth-form"
            >
              <div className="form-group">
                <div className="input-wrapper">
                  <input
                    type="email"
                    id="email"
                    className={`form-control ${errors.email && touched.email ? 'form-control--error' : ''} ${email ? 'has-value' : ''}`}
                    placeholder="Email Address"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    onBlur={() => handleBlur('email')}
                  />
                  <label htmlFor="email" className={`floating-label ${email ? 'floating' : ''}`}>Email Address</label>
                </div>
                {errors.email && touched.email && (
                  <p className="form-error-message">{errors.email}</p>
                )}
              </div>

              <div className="form-group">
                <div className="input-wrapper">
                  <input
                    type={showPassword ? "text" : "password"}
                    id="password"
                    className={`form-control ${errors.password && touched.password ? 'form-control--error' : ''} ${password ? 'has-value' : ''}`}
                    placeholder=" "
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    onBlur={() => handleBlur('password')}
                  />
                  <label htmlFor="password" className={`floating-label ${password ? 'floating' : ''}`}>Password</label>
                  <button type="button" className="password-forgot-link" onClick={(e) => e.preventDefault()}>
                    Forgot?
                  </button>
                  <button
                    type="button"
                    className="password-toggle-button"
                    onClick={() => setShowPassword(!showPassword)}
                    aria-label={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? '🙈' : '👁️'}
                  </button>
                </div>
                {errors.password && touched.password && (
                  <p className="form-error-message">{errors.password}</p>
                )}
              </div>

              <div className="checkbox-group">
                <label className="checkbox-container">
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                  />
                  <span className="checkmark"></span>
                  Remember me
                </label>
              </div>

              <motion.button
                type="submit"
                className="auth-button"
                disabled={isLoading}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {isLoading ? (
                  <span className="loading-spinner">⏳</span>
                ) : (
                  'Sign In'
                )}
              </motion.button>
            </motion.form>

            <motion.div
              className="auth-divider"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              <span>or continue with</span>
            </motion.div>

            <motion.div
              className="social-login"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              <motion.button
                className="social-button google"
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleGoogleSignIn}
              >
                <span>Continue with Google</span>
              </motion.button>
            </motion.div>

            <motion.div
              className="auth-toggle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
            >
              Don't have an account? <Link to="/signup">Sign up</Link>
            </motion.div>
          </div>
        </motion.div>
      </main>

      <style jsx>{`
        .auth-page {
          min-height: 80vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background: linear-gradient(135deg, var(--ifm-color-emphasis-100) 0%, var(--ifm-background-color) 100%);
          padding: 2rem 1rem;
        }

        [data-theme='dark'] .auth-page {
          background: linear-gradient(135deg, var(--ifm-color-emphasis-200) 0%, var(--ifm-background-color) 100%);
        }

        .auth-container {
          width: 100%;
          max-width: 450px;
        }

        .auth-card {
          background: var(--ifm-card-background-color);
          border-radius: 16px;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
          padding: 2.5rem;
          border: 1px solid var(--ifm-color-emphasis-200);
        }

        .auth-header {
          text-align: center;
          margin-bottom: 2rem;
        }

        .auth-header h1 {
          font-size: 1.75rem;
          font-weight: 700;
          margin-bottom: 0.5rem;
          color: var(--ifm-heading-color);
        }

        .auth-header p {
          color: var(--ifm-font-color-base);
          opacity: 0.8;
        }

        .form-group {
          margin-bottom: 1.5rem;
        }

        .input-wrapper {
          position: relative;
          margin-bottom: 1.5rem;
        }

        .form-control {
          width: 100%;
          padding: 16px 80px 16px 16px; /* Increased right padding to accommodate both forgot link and eye icon */
          border: 1px solid var(--ifm-color-emphasis-300);
          border-radius: 8px;
          font-size: 1rem;
          background: var(--ifm-card-background-color);
          color: var(--ifm-font-color-base);
          transition: border-color 0.2s ease, box-shadow 0.2s ease;
          outline: none;
        }

        .form-control:focus {
          border-color: var(--ifm-color-primary);
          box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
        }

        .form-control--error {
          border-color: #e53e3e;
        }

        .form-control.has-value {
          padding-top: 24px;
          padding-bottom: 8px;
        }

        .floating-label {
          position: absolute;
          left: 16px;
          right: 80px; /* Account for the right padding */
          top: 50%;
          transform: translateY(-50%);
          color: var(--ifm-color-emphasis-600);
          pointer-events: none;
          transition: all 0.2s ease;
          font-size: 1rem;
          background: var(--ifm-card-background-color);
          padding: 0 4px;
        }

        .floating-label.floating {
          top: 8px;
          transform: translateY(0);
          font-size: 0.875rem;
          color: var(--ifm-color-primary);
        }

        .form-label {
          font-weight: 600;
          color: var(--ifm-font-color-base);
        }

        .password-forgot-link {
          position: absolute;
          right: 45px; /* Position to the left of the eye icon */
          top: 50%;
          transform: translateY(-50%);
          font-size: 0.875rem;
          color: var(--ifm-color-primary);
          text-decoration: none;
          background: none;
          border: none;
          cursor: pointer;
          padding: 4px 8px;
          z-index: 2;
        }

        .password-toggle-button {
          position: absolute;
          right: 1rem;
          top: 50%;
          transform: translateY(-50%);
          background: none;
          border: none;
          cursor: pointer;
          font-size: 1.2rem;
          padding: 0.25rem;
          color: var(--ifm-font-color-base);
          opacity: 0.6;
          z-index: 3;
        }

        .password-toggle-button:hover {
          opacity: 1;
        }

        .form-error-message {
          color: #e53e3e;
          font-size: 0.875rem;
          margin-top: 0.25rem;
        }

        .checkbox-group {
          margin-bottom: 1.5rem;
        }

        .checkbox-container {
          display: flex;
          align-items: center;
          cursor: pointer;
          font-size: 0.9rem;
          color: var(--ifm-font-color-base);
        }

        .checkbox-container input {
          margin-right: 0.5rem;
        }

        .checkmark {
          position: relative;
          display: inline-block;
          width: 18px;
          height: 18px;
          border: 2px solid var(--ifm-color-emphasis-400);
          border-radius: 4px;
          margin-right: 8px;
          transition: all 0.2s ease;
        }

        .checkbox-container input:checked ~ .checkmark {
          background-color: var(--ifm-color-primary);
          border-color: var(--ifm-color-primary);
        }

        .checkbox-container input:checked ~ .checkmark::after {
          content: '✓';
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          color: white;
          font-size: 0.75rem;
        }

        .auth-button {
          width: 100%;
          padding: 14px;
          background: var(--ifm-color-primary);
          color: white;
          border: none;
          border-radius: 8px;
          font-size: 1rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
          box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
          margin-bottom: 1.5rem;
        }

        .auth-button:hover:not(:disabled) {
          background: var(--ifm-color-primary-dark);
          box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);
        }

        .auth-button:disabled {
          background: var(--ifm-color-emphasis-200);
          cursor: not-allowed;
        }

        .loading-spinner {
          display: inline-block;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .auth-divider {
          text-align: center;
          margin: 2rem 0;
          position: relative;
          color: var(--ifm-color-emphasis-600);
          font-size: 0.9rem;
        }

        .auth-divider::before {
          content: '';
          position: absolute;
          top: 50%;
          left: 0;
          right: 0;
          height: 1px;
          background: var(--ifm-color-emphasis-300);
        }

        .auth-divider span {
          position: relative;
          background: var(--ifm-card-background-color);
          padding: 0 1rem;
        }

        .social-login {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
          margin-bottom: 1.5rem;
        }

        .social-button {
          width: 100%;
          padding: 12px;
          border: 1px solid var(--ifm-color-emphasis-300);
          border-radius: 8px;
          background: var(--ifm-background-color);
          color: var(--ifm-font-color-base);
          font-size: 0.95rem;
          font-weight: 500;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.75rem;
          transition: all 0.2s ease;
        }

        .social-button:hover {
          background: var(--ifm-color-emphasis-100);
          border-color: var(--ifm-color-emphasis-400);
        }

        .social-button.google {
          background: #fff;
          color: #757575;
          border-color: #dadce0;
        }

        .social-button.google:hover {
          background: #f8f9fa;
          border-color: #c6c9ce;
        }

        .auth-toggle {
          text-align: center;
          color: var(--ifm-font-color-base);
          font-size: 0.95rem;
        }

        .auth-toggle a {
          color: var(--ifm-color-primary);
          text-decoration: none;
          font-weight: 600;
        }

        .auth-toggle a:hover {
          text-decoration: underline;
        }
      `}</style>
    </Layout>
  );
}