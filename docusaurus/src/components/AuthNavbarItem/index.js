import React, { useState, useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import Link from '@docusaurus/Link';

const AuthNavbarItem = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState(null);
  const location = useLocation();

  // Check if user is logged in by checking for auth tokens in localStorage
  useEffect(() => {
    const checkAuthStatus = () => {
      // In a real implementation, this would check for BetterAuth tokens
      // For now, we'll simulate by checking a mock token
      const token = localStorage.getItem('better-auth-token');
      if (token) {
        try {
          const userData = JSON.parse(atob(token.split('.')[1])); // Decode JWT payload
          setIsLoggedIn(true);
          setUser(userData);
        } catch (e) {
          setIsLoggedIn(false);
          setUser(null);
        }
      } else {
        setIsLoggedIn(false);
        setUser(null);
      }
    };

    checkAuthStatus();

    // Listen for storage changes (in case login/logout happens in another tab)
    const handleStorageChange = () => {
      checkAuthStatus();
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [location.pathname]);

  if (isLoggedIn && user) {
    // User is logged in - show user menu
    return (
      <div className="navbar__item auth-navbar-item">
        <div className="dropdown dropdown--right dropdown--username">
          <button className="navbar__link dropdown__trigger">
            <span className="username-display">{user.name || user.email?.split('@')[0]}</span>
          </button>
          <ul className="dropdown__menu">
            <li>
              <Link to="/dashboard" className="dropdown__link">
                Dashboard
              </Link>
            </li>
            <li>
              <Link to="/profile" className="dropdown__link">
                Profile
              </Link>
            </li>
            <li>
              <a
                href="#"
                className="dropdown__link"
                onClick={(e) => {
                  e.preventDefault();
                  // In a real implementation, this would call BetterAuth's signout API
                  localStorage.removeItem('better-auth-token');
                  window.dispatchEvent(new Event('storage'));
                  window.location.href = '/signin';
                }}
              >
                Logout
              </a>
            </li>
          </ul>
        </div>
      </div>
    );
  } else {
    // User is not logged in - show sign in/up buttons
    return (
      <div className="navbar__item auth-navbar-item">
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