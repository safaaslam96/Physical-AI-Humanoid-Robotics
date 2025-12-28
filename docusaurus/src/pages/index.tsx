import type {ReactNode} from 'react';
// @ts-ignore: no type declarations for '@docusaurus/useDocusaurusContext'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
// @ts-ignore: no type declarations for '@theme/Layout'
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import AIChatPopup from '@site/src/components/AIChatPopup/AIChatPopup';

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics`}
      description="Complete educational resource for Physical AI and Humanoid Robotics. AI Systems in the Physical World. Embodied Intelligence.">
      <main>
        <HomepageFeatures />
      </main>
      <AIChatPopup />
    </Layout>
  );
}
