import React from 'react';
import AIChatPopup from '../components/AIChatPopup/AIChatPopup';

interface RootLayoutProps {
  children: React.ReactNode;
}

const RootLayout: React.FC<RootLayoutProps> = ({ children }) => {
  return (
    <>
      {children}
      <AIChatPopup />
    </>
  );
};

export default RootLayout;