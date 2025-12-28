import React, { useState } from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import { motion } from 'framer-motion';

export default function Signup() {
  const { siteConfig } = useDocusaurusContext();

  // Basic sign up form state
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [agreeToTerms, setAgreeToTerms] = useState(false);

  // Personalization questions state
  const [softwareBackground, setSoftwareBackground] = useState('');
  const [hasHighEndGPU, setHasHighEndGPU] = useState('');
  const [familiarWithROS2, setFamiliarWithROS2] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});
  const [currentStep, setCurrentStep] = useState(1); // 1 = basic info, 2 = personalization, 3 = success

  // Validate individual fields
  const validateField = (name, value) => {
    switch (name) {
      case 'name':
        if (!value) return 'Name is required';
        return '';
      case 'email':
        if (!value) return 'Email is required';
        if (!/\S+@\S+\.\S+/.test(value)) return 'Email address is invalid';
        return '';
      case 'password':
        if (!value) return 'Password is required';
        if (value.length < 6) return 'Password must be at least 6 characters';
        return '';
      case 'confirmPassword':
        if (!value) return 'Please confirm your password';
        if (value !== password) return 'Passwords do not match';
        return '';
      case 'terms':
        if (!value) return 'You must agree to the terms and conditions';
        return '';
      case 'softwareBackground':
        if (!value) return 'Please select your software experience level';
        return '';
      case 'hasHighEndGPU':
        if (!value) return 'Please answer if you have access to high-end GPU';
        return '';
      case 'familiarWithROS2':
        if (!value) return 'Please answer if you are familiar with ROS 2';
        return '';
      default:
        return '';
    }
  };

  // Validate basic form
  const validateBasicForm = () => {
    const newErrors = {};
    newErrors.name = validateField('name', name);
    newErrors.email = validateField('email', email);
    newErrors.password = validateField('password', password);
    newErrors.confirmPassword = validateField('confirmPassword', confirmPassword);
    newErrors.terms = validateField('terms', agreeToTerms);

    setErrors(newErrors);
    return !Object.values(newErrors).some(error => error !== '');
  };

  // Validate personalization form
  const validatePersonalizationForm = () => {
    const newErrors = {};
    newErrors.softwareBackground = validateField('softwareBackground', softwareBackground);
    newErrors.hasHighEndGPU = validateField('hasHighEndGPU', hasHighEndGPU);
    newErrors.familiarWithROS2 = validateField('familiarWithROS2', familiarWithROS2);

    setErrors(newErrors);
    return !Object.values(newErrors).some(error => error !== '');
  };

  const handleBasicSubmit = (e) => {
    e.preventDefault();
    if (validateBasicForm()) {
      setCurrentStep(2); // Move to personalization questions
    }
  };

  const handlePersonalizationSubmit = async (e) => {
    e.preventDefault();
    if (validatePersonalizationForm()) {
      setIsLoading(true);
      try {
        // Simulate API call to create user with personalization data
        await new Promise(resolve => setTimeout(resolve, 1500));
        console.log('User created with profile:', {
          name,
          email,
          softwareBackground,
          hasHighEndGPU,
          familiarWithROS2
        });
        setCurrentStep(3); // Move to success screen
      } catch (error) {
        console.error('Sign up error:', error);
        alert('Account creation failed. Please try again.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleBlur = (fieldName) => {
    setTouched({ ...touched, [fieldName]: true });
    setErrors({
      ...errors,
      [fieldName]: validateField(fieldName,
        fieldName === 'email' ? email :
        fieldName === 'password' ? password :
        fieldName === 'confirmPassword' ? confirmPassword :
        fieldName === 'name' ? name :
        fieldName === 'terms' ? agreeToTerms :
        fieldName === 'softwareBackground' ? softwareBackground :
        fieldName === 'hasHighEndGPU' ? hasHighEndGPU :
        fieldName === 'familiarWithROS2' ? familiarWithROS2 : ''
      )
    });
  };

  // Handle Google OAuth sign up
  const handleGoogleSignUp = () => {
    // In a real implementation, this would redirect to BetterAuth's Google OAuth endpoint
    console.log('Initiating Google OAuth sign up');
  };

  if (currentStep === 3) {
    return (
      <Layout title={`Sign Up - ${siteConfig.title}`} description="Create your Physical AI & Humanoid Robotics account">
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
                  Account Created Successfully!
                </motion.h1>
                <motion.p
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  Welcome to the Physical AI & Humanoid Robotics community
                </motion.p>
              </div>

              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="success-content"
              >
                <div className="success-icon">🎉</div>
                <p className="success-message">
                  Your account has been created successfully. We've saved your background information to personalize your learning experience.
                </p>
              </motion.div>

              <motion.div
                className="auth-toggle"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
              >
                <Link to="/signin" className="auth-button">Go to Sign In</Link>
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
            text-align: center;
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

          .success-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
          }

          .success-message {
            font-size: 1.1rem;
            color: var(--ifm-font-color-base);
            margin-bottom: 2rem;
            line-height: 1.6;
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
          }

          .auth-button:hover:not(:disabled) {
            background: var(--ifm-color-primary-dark);
            box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);
          }
        `}</style>
      </Layout>
    );
  }

  if (currentStep === 2) {
    return (
      <Layout title={`Sign Up - ${siteConfig.title}`} description="Complete your profile to personalize your experience">
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
                  Personalize Your Experience
                </motion.h1>
                <motion.p
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  Answer a few questions to customize your learning journey
                </motion.p>
              </div>

              <motion.form
                onSubmit={handlePersonalizationSubmit}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="auth-form"
              >
                <div className="form-group">
                  <div className="input-wrapper">
                    <select
                      id="softwareBackground"
                      className={`form-control ${errors.softwareBackground && touched.softwareBackground ? 'form-control--error' : ''} ${softwareBackground ? 'has-value' : ''}`}
                      value={softwareBackground}
                      onChange={(e) => setSoftwareBackground(e.target.value)}
                      onBlur={() => handleBlur('softwareBackground')}
                    >
                      <option value="">Select your software experience</option>
                      <option value="beginner">Beginner</option>
                      <option value="intermediate">Intermediate</option>
                      <option value="advanced">Advanced</option>
                    </select>
                    <label htmlFor="softwareBackground" className={`floating-label ${softwareBackground ? 'floating' : ''}`}>What is your software experience?</label>
                  </div>
                  {errors.softwareBackground && touched.softwareBackground && (
                    <p className="form-error-message">{errors.softwareBackground}</p>
                  )}
                </div>

                <div className="form-group">
                  <div className="input-wrapper">
                    <select
                      id="hasHighEndGPU"
                      className={`form-control ${errors.hasHighEndGPU && touched.hasHighEndGPU ? 'form-control--error' : ''} ${hasHighEndGPU ? 'has-value' : ''}`}
                      value={hasHighEndGPU}
                      onChange={(e) => setHasHighEndGPU(e.target.value)}
                      onBlur={() => handleBlur('hasHighEndGPU')}
                    >
                      <option value="">Select option</option>
                      <option value="yes">Yes</option>
                      <option value="no">No</option>
                    </select>
                    <label htmlFor="hasHighEndGPU" className={`floating-label ${hasHighEndGPU ? 'floating' : ''}`}>Do you have access to high-end GPU (RTX 4070 or better)?</label>
                  </div>
                  {errors.hasHighEndGPU && touched.hasHighEndGPU && (
                    <p className="form-error-message">{errors.hasHighEndGPU}</p>
                  )}
                </div>

                <div className="form-group">
                  <div className="input-wrapper">
                    <select
                      id="familiarWithROS2"
                      className={`form-control ${errors.familiarWithROS2 && touched.familiarWithROS2 ? 'form-control--error' : ''} ${familiarWithROS2 ? 'has-value' : ''}`}
                      value={familiarWithROS2}
                      onChange={(e) => setFamiliarWithROS2(e.target.value)}
                      onBlur={() => handleBlur('familiarWithROS2')}
                    >
                      <option value="">Select option</option>
                      <option value="yes">Yes</option>
                      <option value="no">No</option>
                    </select>
                    <label htmlFor="familiarWithROS2" className={`floating-label ${familiarWithROS2 ? 'floating' : ''}`}>Are you familiar with ROS 2?</label>
                  </div>
                  {errors.familiarWithROS2 && touched.familiarWithROS2 && (
                    <p className="form-error-message">{errors.familiarWithROS2}</p>
                  )}
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
                    'Complete Sign Up'
                  )}
                </motion.button>
              </motion.form>

              <motion.div
                className="auth-toggle"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
              >
                <button
                  type="button"
                  onClick={() => setCurrentStep(1)}
                  className="back-button"
                >
                  ← Back to Account Info
                </button>
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

          .form-label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: var(--ifm-font-color-base);
          }

          .input-wrapper {
            position: relative;
            margin-bottom: 1.5rem;
          }

          .form-control {
            width: 100%;
            padding: 16px 16px 16px 16px;
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

          .form-control.has-value {
            padding-top: 24px;
            padding-bottom: 8px;
          }

          .floating-label {
            position: absolute;
            left: 16px;
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

          .form-error-message {
            color: #e53e3e;
            font-size: 0.875rem;
            margin-top: 0.25rem;
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

          .auth-toggle {
            text-align: center;
            color: var(--ifm-font-color-base);
            font-size: 0.95rem;
          }

          .back-button {
            background: none;
            border: none;
            color: var(--ifm-color-primary);
            cursor: pointer;
            font-size: 0.95rem;
            text-decoration: none;
            font-weight: 600;
          }

          .back-button:hover {
            text-decoration: underline;
          }
        `}</style>
      </Layout>
    );
  }

  // Step 1: Basic sign up form
  return (
    <Layout title={`Sign Up - ${siteConfig.title}`} description="Create your Physical AI & Humanoid Robotics account">
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
                Create Account
              </motion.h1>
              <motion.p
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                Join the Physical AI & Humanoid Robotics community
              </motion.p>
            </div>

            <motion.form
              onSubmit={handleBasicSubmit}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="auth-form"
            >
              <div className="form-group">
                <div className="input-wrapper">
                  <input
                    type="text"
                    id="name"
                    className={`form-control ${errors.name && touched.name ? 'form-control--error' : ''} ${name ? 'has-value' : ''}`}
                    placeholder=" "
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    onBlur={() => handleBlur('name')}
                  />
                  <label htmlFor="name" className={`floating-label ${name ? 'floating' : ''}`}>Full Name</label>
                </div>
                {errors.name && touched.name && (
                  <p className="form-error-message">{errors.name}</p>
                )}
              </div>

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

              <div className="form-group">
                <div className="input-wrapper">
                  <input
                    type={showConfirmPassword ? "text" : "password"}
                    id="confirmPassword"
                    className={`form-control ${errors.confirmPassword && touched.confirmPassword ? 'form-control--error' : ''} ${confirmPassword ? 'has-value' : ''}`}
                    placeholder=" "
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    onBlur={() => handleBlur('confirmPassword')}
                  />
                  <label htmlFor="confirmPassword" className={`floating-label ${confirmPassword ? 'floating' : ''}`}>Confirm Password</label>
                  <button
                    type="button"
                    className="password-toggle-button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    aria-label={showConfirmPassword ? "Hide password" : "Show password"}
                  >
                    {showConfirmPassword ? '🙈' : '👁️'}
                  </button>
                </div>
                {errors.confirmPassword && touched.confirmPassword && (
                  <p className="form-error-message">{errors.confirmPassword}</p>
                )}
              </div>

              <div className="checkbox-group">
                <label className="checkbox-container">
                  <input
                    type="checkbox"
                    checked={agreeToTerms}
                    onChange={(e) => setAgreeToTerms(e.target.checked)}
                    onBlur={() => handleBlur('terms')}
                  />
                  <span className="checkmark"></span>
                  I agree to the Terms of Service and Privacy Policy
                </label>
                {errors.terms && touched.terms && (
                  <p className="form-error-message">{errors.terms}</p>
                )}
              </div>

              <motion.button
                type="submit"
                className="auth-button"
                disabled={isLoading}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Continue
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
                onClick={handleGoogleSignUp}
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
              Already have an account? <Link to="/signin">Sign in</Link>
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

        .form-label {
          display: block;
          margin-bottom: 0.5rem;
          font-weight: 600;
          color: var(--ifm-font-color-base);
        }

        .input-wrapper {
          position: relative;
          margin-bottom: 1.5rem;
        }

        .form-control {
          width: 100%;
          padding: 16px 16px 16px 16px;
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

        .form-control.has-value {
          padding-top: 24px;
          padding-bottom: 8px;
        }

        .floating-label {
          position: absolute;
          left: 16px;
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