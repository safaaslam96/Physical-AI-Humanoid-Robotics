import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  id: string;
  email: string;
  name?: string;
  softwareBackground?: string;
  hasHighEndGPU?: string;
  familiarWithROS2?: string;
  [key: string]: any;
}

interface AuthContextType {
  user: User | null;
  status: 'authenticated' | 'unauthenticated' | 'loading';
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  signUp: (userData: { name: string; email: string; password: string; [key: string]: any }) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [status, setStatus] = useState<'authenticated' | 'unauthenticated' | 'loading'>('loading');

  useEffect(() => {
    // Check if user is logged in by checking for tokens in localStorage
    const token = localStorage.getItem('better-auth-token');
    if (token) {
      // In a real implementation, you would verify the token with the server
      // For now, we'll just check if it exists
      fetch('/api/auth/session', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      .then(response => {
        if (response.ok) {
          return response.json();
        }
        throw new Error('Invalid token');
      })
      .then(userData => {
        setUser(userData);
        setStatus('authenticated');
      })
      .catch(() => {
        setStatus('unauthenticated');
      });
    } else {
      setStatus('unauthenticated');
    }
  }, []);

  const signIn = async (email: string, password: string) => {
    setStatus('loading');
    try {
      const response = await fetch('/api/auth/signin', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('better-auth-token', data.token);
        setUser(data.user);
        setStatus('authenticated');
      } else {
        throw new Error('Sign in failed');
      }
    } catch (error) {
      setStatus('unauthenticated');
      throw error;
    }
  };

  const signOut = async () => {
    setStatus('loading');
    try {
      await fetch('/api/auth/signout', {
        method: 'POST',
      });
      localStorage.removeItem('better-auth-token');
      setUser(null);
      setStatus('unauthenticated');
    } catch (error) {
      console.error('Sign out error:', error);
      setStatus('unauthenticated');
    }
  };

  const signUp = async (userData: { name: string; email: string; password: string; [key: string]: any }) => {
    setStatus('loading');
    try {
      const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('better-auth-token', data.token);
        setUser(data.user);
        setStatus('authenticated');
      } else {
        throw new Error('Sign up failed');
      }
    } catch (error) {
      setStatus('unauthenticated');
      throw error;
    }
  };

  return (
    <AuthContext.Provider value={{ user, status, signIn, signOut, signUp }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};