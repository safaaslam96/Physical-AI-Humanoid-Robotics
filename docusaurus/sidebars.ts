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
  // Clean, professional sidebar for the Physical AI & Humanoid Robotics book
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        {
          type: 'category',
          label: 'Chapter 1: Foundations of Physical AI and Embodied Intelligence',
          items: [
            'part1/chapter1'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 2: Digital AI to Robots Understanding Physical Laws',
          items: [
            'part1/chapter2'
          ],
          collapsed: false
        }
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        {
          type: 'category',
          label: 'Chapter 3: Overview of Humanoid Robotics Landscape',
          items: [
            'part2/chapter3'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 4: Sensor Systems (LIDAR, Cameras, IMUs, Force/Torque Sensors)',
          items: [
            'part2/chapter4'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 5: ROS 2 Architecture and Core Concepts',
          items: [
            'part2/chapter5'
          ],
          collapsed: false
        }
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        {
          type: 'category',
          label: 'Chapter 6: Nodes, Topics, Services, and Actions',
          items: [
            'part3/chapter6'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 7: Building ROS 2 Packages with Python',
          items: [
            'part3/chapter7'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 8: Launch Files and Parameter Management',
          items: [
            'part3/chapter8'
          ],
          collapsed: false
        }
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        {
          type: 'category',
          label: 'Chapter 9: Understanding URDF for Humanoids',
          items: [
            'part4/chapter9'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 10: Gazebo Simulation Environment Setup',
          items: [
            'part4/chapter10'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 11: Physics Simulation Principles',
          items: [
            'part4/chapter11'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 12: URDF and SDF Robot Description Formats',
          items: [
            'part4/chapter12'
          ],
          collapsed: false
        }
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Module 5: Advanced Robotics',
      items: [
        {
          type: 'category',
          label: 'Chapter 13: Physics Simulation and Sensor Simulation',
          items: [
            'part5/chapter13'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 14: Introduction to Unity for Robot Visualization',
          items: [
            'part5/chapter14'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 15: NVIDIA Isaac SDK and Isaac Sim',
          items: [
            'part5/chapter15'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 16: AI-Powered Perception and Manipulation',
          items: [
            'part5/chapter16'
          ],
          collapsed: false
        }
      ],
      collapsed: false
    },
    {
      type: 'category',
      label: 'Module 6: Humanoid Development',
      items: [
        {
          type: 'category',
          label: 'Chapter 17: Isaac ROS and Visual SLAM',
          items: [
            'part6/chapter17'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 18: Advanced Perception Techniques',
          items: [
            'part6/chapter18'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 19: Path Planning for Bipedal Humanoid Movement',
          items: [
            'part6/chapter19'
          ],
          collapsed: false
        },
        {
          type: 'category',
          label: 'Chapter 20: Sim-to-Real Transfer Techniques',
          items: [
            'part6/chapter20'
          ],
          collapsed: false
        }
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
