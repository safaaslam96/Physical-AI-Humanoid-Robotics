// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'AI Systems in the Physical World. Embodied Intelligence.',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-book-domain.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub Pages: https://<USERNAME>.github.io/<REPO>/
  baseUrl: '/Physical-AI-Humanoid-Robotics/',

  // GitHub pages deployment config.
  organizationName: 'your-org', // Usually your GitHub org/user name.
  projectName: 'Physical-AI-Humanoid-Robotics', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI Logo',
          src: 'img/logo.svg',
        },
        items: [
          {to: '/', label: 'Home', position: 'left'},
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book',
          },
          {to: '/chatbot', label: 'AI Chatbot', position: 'left'},
          {to: '/personalization', label: 'Personalize', position: 'left'},
          {to: '/translation', label: 'Translate', position: 'left'},
          {to: '/blog', label: 'Blog', position: 'left'},
          {
            href: 'https://github.com/facebook/docusaurus',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: '📚 Book Chapters',
            items: [
              {
                label: 'Introduction to Physical AI',
                to: '/docs/part1/chapter1',
              },
              {
                label: 'ROS 2 Architecture',
                to: '/docs/part2/chapter3',
              },
              {
                label: 'Simulation Environments',
                to: '/docs/part3/chapter6',
              },
              {
                label: 'NVIDIA Isaac SDK',
                to: '/docs/part4/chapter9',
              },
              {
                label: 'Humanoid Development',
                to: '/docs/part5/chapter13',
              },
              {
                label: 'LLM Integration',
                to: '/docs/part6/chapter17',
              },
            ],
          },
          {
            title: '🤖 Core Systems',
            items: [
              {
                label: 'ROS 2 Integration',
                href: '/docs/part2/chapter3',
              },
              {
                label: 'Isaac Sim',
                href: '/docs/part4/chapter10',
              },
              {
                label: 'Computer Vision',
                href: '/docs/part4/chapter11',
              },
              {
                label: 'Bipedal Locomotion',
                href: '/docs/part5/chapter14',
              },
              {
                label: 'Natural Interaction',
                href: '/docs/part6/chapter16',
              },
              {
                label: 'Cognitive Planning',
                href: '/docs/part6/chapter19',
              },
            ],
          },
          {
            title: '🔗 Resources',
            items: [
              {
                label: 'Documentation',
                to: '/docs/intro',
              },
              {
                label: 'AI Chatbot',
                to: '/chatbot',
              },
              {
                label: 'Code Examples',
                href: 'https://github.com/your-org/Physical-AI-Humanoid-Robotics/tree/main/code-examples',
              },
              {
                label: 'Simulation Setup',
                href: '/docs/part3/chapter6',
              },
              {
                label: 'Hardware Guide',
                href: '/docs/part5/chapter15',
              },
              {
                label: 'Safety Guidelines',
                href: '/docs/part6/chapter20',
              },
            ],
          },
          {
            title: '👥 Community',
            items: [
              {
                label: 'GitHub Repository',
                href: 'https://github.com/your-org/Physical-AI-Humanoid-Robotics',
              },
              {
                label: 'Discord Community',
                href: 'https://discord.gg/physical-ai',
              },
              {
                label: 'Research Papers',
                href: '/docs/part6/chapter21',
              },
              {
                label: 'Tutorials',
                to: '/docs/intro',
              },
              {
                label: 'Contributing Guide',
                href: 'https://github.com/your-org/Physical-AI-Humanoid-Robotics/blob/main/CONTRIBUTING.md',
              },
              {
                label: 'Issue Tracker',
                href: 'https://github.com/your-org/Physical-AI-Humanoid-Robotics/issues',
              },
            ],
          },
          {
            title: '📰 Updates',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'Release Notes',
                href: 'https://github.com/your-org/Physical-AI-Humanoid-Robotics/releases',
              },
              {
                label: 'Newsletter',
                href: '#',
              },
              {
                label: 'Events & Workshops',
                href: '/events',
              },
              {
                label: 'Roadmap',
                href: 'https://github.com/your-org/Physical-AI-Humanoid-Robotics/projects',
              },
              {
                label: 'Status Page',
                href: 'https://status.physical-ai.com',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Educational Project. All rights reserved. | Empowering the future of embodied artificial intelligence.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;