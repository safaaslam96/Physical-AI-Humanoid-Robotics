import React from 'react';
import Layout from '@theme/Layout';
import ChatbotPopup from '../components/Chatbot/ChatbotPopup';

interface RootLayoutProps {
  children: React.ReactNode;
}

const RootLayout: React.FC<RootLayoutProps> = ({ children }) => {
  return (
    <Layout>
      {children}
      <ChatbotPopup />
    </Layout>
  );
};

export default RootLayout;