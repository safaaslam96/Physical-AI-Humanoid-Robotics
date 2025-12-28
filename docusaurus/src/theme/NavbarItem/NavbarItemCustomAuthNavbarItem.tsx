import React from 'react';
import NavbarItem from '@theme/NavbarItem';
import Link from '@docusaurus/Link';
import { useAuth } from '../../contexts/AuthContext';

const NavbarItemCustomAuthNavbarItem = (props) => {
  const { user, status } = useAuth();

  if (status === 'authenticated' && user) {
    // User is authenticated - show dashboard/logout
    return (
      <div className="navbar__item auth-navbar-item">
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

export default NavbarItemCustomAuthNavbarItem;