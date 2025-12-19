import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatbotPopup from '../components/Chatbot/ChatbotPopup';

type Props = {
  children?: React.ReactNode;
  [key: string]: unknown;
};

const LayoutWrapper = (props: Props): JSX.Element => {
  return (
    <>
      <OriginalLayout {...props}>{props.children}</OriginalLayout>
      <ChatbotPopup />
    </>
  );
};

export default LayoutWrapper;