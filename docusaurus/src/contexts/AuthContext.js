import React, { createContext, useContext, useReducer, useEffect } from 'react';

// Define the initial state
const initialState = {
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: true,
};

// Define the reducer
const authReducer = (state, action) => {
  switch (action.type) {
    case 'LOGIN_START':
      return {
        ...state,
        isLoading: true,
      };
    case 'LOGIN_SUCCESS':
      return {
        ...state,
        user: action.payload.user,
        token: action.payload.token,
        isAuthenticated: true,
        isLoading: false,
      };
    case 'LOGIN_FAILURE':
      return {
        ...state,
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
      };
    case 'LOGOUT':
      return {
        ...state,
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
      };
    case 'SET_USER':
      return {
        ...state,
        user: action.payload,
        isAuthenticated: !!action.payload,
        isLoading: false,
      };
    default:
      return state;
  }
};

// Create the context
const AuthContext = createContext(undefined);

// AuthProvider component
export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Check for existing token on app load
  useEffect(() => {
    const token = localStorage.getItem('better-auth-token');
    const user = localStorage.getItem('better-auth-user');

    if (token && user) {
      try {
        const parsedUser = JSON.parse(user);
        dispatch({
          type: 'SET_USER',
          payload: parsedUser,
        });
      } catch (error) {
        console.error('Error parsing user data:', error);
        // Clear invalid data
        localStorage.removeItem('better-auth-token');
        localStorage.removeItem('better-auth-user');
        dispatch({ type: 'LOGIN_FAILURE' });
      }
    } else {
      dispatch({ type: 'LOGIN_FAILURE' });
    }
  }, []);

  // Login function
  const login = async (email, password) => {
    dispatch({ type: 'LOGIN_START' });

    try {
      // In a real implementation, this would call BetterAuth's login API
      // For now, we'll simulate a successful login

      // Mock user data
      const mockUser = {
        id: 'mock-user-id',
        email,
        name: email.split('@')[0], // Use part of email as name
        createdAt: new Date().toISOString(),
      };

      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Store in localStorage (simulating JWT token)
      const mockToken = btoa(JSON.stringify({ id: mockUser.id, email }));
      localStorage.setItem('better-auth-token', mockToken);
      localStorage.setItem('better-auth-user', JSON.stringify(mockUser));

      dispatch({
        type: 'LOGIN_SUCCESS',
        payload: {
          user: mockUser,
          token: mockToken,
        },
      });

      return { success: true, user: mockUser };
    } catch (error) {
      dispatch({ type: 'LOGIN_FAILURE' });
      return { success: false, error: error.message };
    }
  };

  // Logout function
  const logout = () => {
    localStorage.removeItem('better-auth-token');
    localStorage.removeItem('better-auth-user');
    dispatch({ type: 'LOGOUT' });
  };

  // Register function
  const register = async (userData) => {
    dispatch({ type: 'LOGIN_START' });

    try {
      // In a real implementation, this would call BetterAuth's register API
      // For now, we'll simulate a successful registration

      // Mock user data with profile info
      const mockUser = {
        id: `user-${Date.now()}`,
        email: userData.email,
        name: userData.name,
        softwareBackground: userData.softwareBackground,
        hasHighEndGPU: userData.hasHighEndGPU,
        familiarWithROS2: userData.familiarWithROS2,
        createdAt: new Date().toISOString(),
      };

      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Store in localStorage (simulating JWT token)
      const mockToken = btoa(JSON.stringify({ id: mockUser.id, email: mockUser.email }));
      localStorage.setItem('better-auth-token', mockToken);
      localStorage.setItem('better-auth-user', JSON.stringify(mockUser));

      dispatch({
        type: 'LOGIN_SUCCESS',
        payload: {
          user: mockUser,
          token: mockToken,
        },
      });

      return { success: true, user: mockUser };
    } catch (error) {
      dispatch({ type: 'LOGIN_FAILURE' });
      return { success: false, error: error.message };
    }
  };

  // Value to be provided to consumers
  const value = {
    ...state,
    login,
    logout,
    register,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Custom hook to use the auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};