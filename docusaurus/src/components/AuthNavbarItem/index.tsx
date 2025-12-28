import React from 'react';
import Link from '@docusaurus/Link';
import { useAuth } from '@docusaurus/auth/client';

const AuthNavbarItem = () => {
  const { data: session, status } = useAuth();

  if (status === 'authenticated' && session) {
    // User is authenticated - show dashboard/logout
    return (
      <div className="auth-navbar-item">
        <Link
          to="/dashboard"
          className="button button--primary button--sm auth-button-style"
          style={{ marginRight: '0.5rem' }}
        >
          Dashboard
        </Link>
        <Link
          to="/api/auth/signout"
          className="button button--secondary button--sm auth-button-style"
        >
          Logout
        </Link>
      </div>
    );
  } else {
    // User is not authenticated - show sign in/up
    return (
      <div className="auth-navbar-item">
        <Link
          to="/signin"
          className="button button--primary button--sm auth-button-style"
          style={{ marginRight: '0.5rem' }}
        >
          Sign In
        </Link>
        <Link
          to="/signup"
          className="button button--secondary button--sm auth-button-style"
        >
          Sign Up
        </Link>
      </div>
    );
  }
};

export default AuthNavbarItem;