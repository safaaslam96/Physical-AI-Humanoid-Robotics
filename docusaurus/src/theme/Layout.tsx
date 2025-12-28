import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import AIChatPopup from '../components/AIChatPopup/AIChatPopup';
import { AuthProvider } from '../contexts/AuthContext';

type Props = {
  children?: React.ReactNode;
  [key: string]: unknown;
};

const LayoutWrapper = (props: Props): JSX.Element => {
  return (
    <AuthProvider>
      <OriginalLayout {...props}>{props.children}</OriginalLayout>
      <AIChatPopup />
    </AuthProvider>
  );
};

export default LayoutWrapper;