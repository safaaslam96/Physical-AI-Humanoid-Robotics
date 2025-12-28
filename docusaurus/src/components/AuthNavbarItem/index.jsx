import React from 'react';
import Link from '@docusaurus/Link';
import { useAuth } from '../../contexts/AuthContext';

const AuthNavbarItem = (props) => {
  const { user, status } = useAuth();

  // Style to position the auth buttons in the navbar
  const authButtonsStyle = {
    position: 'absolute',
    right: '20px',
    top: '50%',
    transform: 'translateY(-50%)',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem'
  };

  if (status === 'authenticated' && user) {
    // User is authenticated - show dashboard/logout
    return (
      <div className="auth-navbar-container" style={authButtonsStyle}>
        <Link
          to="/dashboard"
          className="button button--primary button--sm margin-right--sm"
        >
          Dashboard
        </Link>
        <Link
          to="/api/auth/signout"
          className="button button--secondary button--sm"
        >
          Logout
        </Link>
      </div>
    );
  } else {
    // User is not authenticated - show sign in/up
    return (
      <div className="auth-navbar-container" style={authButtonsStyle}>
        <Link
          to="/signin"
          className="button button--primary button--sm margin-right--sm"
        >
          Sign In
        </Link>
        <Link
          to="/signup"
          className="button button--secondary button--sm"
        >
          Sign Up
        </Link>
      </div>
    );
  }
};

export default AuthNavbarItem;