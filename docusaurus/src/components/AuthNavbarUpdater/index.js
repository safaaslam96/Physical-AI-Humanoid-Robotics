/**
 * Global authentication navbar updater
 * Updates the navbar buttons based on authentication state
 */

import { useAuth } from '@docusaurus/auth/client';
import { useEffect } from 'react';

const AuthNavbarUpdater = () => {
  const { data: session, status } = useAuth();

  useEffect(() => {
    const updateAuthButtons = () => {
      const placeholder = document.querySelector('.auth-buttons-placeholder');
      if (!placeholder) return;

      if (status === 'authenticated' && session) {
        // User is authenticated - show dashboard/logout
        placeholder.innerHTML = `
          <a href="/dashboard" class="button button--primary button--sm margin-right--sm">Dashboard</a>
          <a href="/api/auth/signout" class="button button--secondary button--sm">Logout</a>
        `;
      } else {
        // User is not authenticated - show sign in/up
        placeholder.innerHTML = `
          <a href="/signin" class="button button--primary button--sm margin-right--sm">Sign In</a>
          <a href="/signup" class="button button--secondary button--sm">Sign Up</a>
        `;
      }
    };

    // Wait for the DOM to be ready before updating
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', updateAuthButtons);
    } else {
      updateAuthButtons();
    }
  }, [status, session]);

  return null;
};

export default AuthNavbarUpdater;