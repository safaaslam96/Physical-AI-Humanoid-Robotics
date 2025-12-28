import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Complete educational resource for AI Systems in the Physical World. Embodied Intelligence.',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://physical-ai-humanoid-robotics-wine.vercel.app',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For Vercel deployment, baseUrl should be '/'
  baseUrl: '/',
  trailingSlash: false,

  // GitHub pages deployment config.
  // Not using GitHub pages - deploying to Vercel
  organizationName: 'safaaslam96', // GitHub username
  projectName: 'Physical-AI-Humanoid-Robotics', // Repo name

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
    localeConfigs: {
      ur: {
        label: 'اردو',
        direction: 'rtl',
        htmlLang: 'ur',
        path: 'ur'
      }
    },
  },

  plugins: [
    './src/plugins/text-highlighter',
  ],
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],


  scripts: [],

  stylesheets: [
    {
      href: 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
      type: 'text/css',
      rel: 'stylesheet',
    },
    {
      href: 'https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;500;600;700&display=swap',
      type: 'text/css',
      rel: 'stylesheet',
    },
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'light', // Changed to light mode by default
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    fonts: {
      body: ['"Noto Nastaliq Urdu"', 'sans-serif'],
    },
 navbar: {
  title: 'Physical AI & Humanoid Robotics',
  items: [
    {
      type: 'localeDropdown',
      position: 'right',
    },
    {
      to: '/signin',
      label: 'Sign In',
      position: 'right',
      className: 'auth-button signin-button',
    },
    {
      to: '/signup',
      label: 'Sign Up',
      position: 'right',
      className: 'auth-button signup-button',
    },
  ],
},
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {
              label: 'Book Chapters',
              to: '/docs/intro',
            },
            {
              label: 'Part I: Introduction to Physical AI',
              to: '/docs/part1/chapter1',
            },
            {
              label: 'Part II: The Robotic Nervous System',
              to: '/docs/part2/chapter3',
            },
            {
              label: 'Modules Overview',
              to: '/docs/part1/chapter2',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/facebook/docusaurus/discussions',
            },
            {
              label: 'Discord',
              href: 'https://discordapp.com/invite/docusaurus',
            },
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/physical-ai-humanoid-robotics',
            },
            {
              label: 'Research Papers',
              href: 'https://scholar.google.com',
            },
          ],
        },
        {
          title: 'About',
          items: [
            {
              label: 'Our Mission',
              to: '/docs/intro',
            },
            {
              label: 'Physical AI & Robotics',
              to: '/docs/part1/chapter1',
            },
            {
              label: 'Humanoid Development',
              to: '/docs/part1/chapter2',
            },
            {
              label: 'Contact Us',
              href: 'mailto:info@physicalai.com',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
