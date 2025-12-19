import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar for the Physical AI & Humanoid Robotics book
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'part1/chapter1',
        'part1/chapter2'
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'ROS 2 and Core Concepts',
      items: [
        'part2/chapter3',
        'part2/chapter4',
        'part2/chapter5'
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Simulation and Modeling',
      items: [
        'part3/chapter6',
        'part3/chapter7',
        'part3/chapter8'
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Advanced Perception and Planning',
      items: [
        'part4/chapter9',
        'part4/chapter10',
        'part4/chapter11',
        'part4/chapter12'
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Humanoid Control and Interaction',
      items: [
        'part5/chapter13',
        'part5/chapter14',
        'part5/chapter15'
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'AI Integration and Autonomy',
      items: [
        'part6/chapter16',
        'part6/chapter17',
        'part6/chapter18',
        'part6/chapter19',
        'part6/chapter20'
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/appendix_a',
        'appendices/appendix_b',
        'appendices/appendix_c'
      ],
      collapsed: false
    }
  ],
};

export default sidebars;
