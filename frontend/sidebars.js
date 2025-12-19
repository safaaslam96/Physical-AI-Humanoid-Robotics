// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Part 1: Introduction to Physical AI',
      items: [
        'part1/chapter1',
        'part1/chapter2'
      ],
    },
    {
      type: 'category',
      label: 'Part 2: The Robotic Nervous System (ROS 2)',
      items: [
        'part2/chapter3',
        'part2/chapter4',
        'part2/chapter5'
      ],
    },
    {
      type: 'category',
      label: 'Part 3: The Digital Twin (Gazebo & Unity)',
      items: [
        'part3/chapter6',
        'part3/chapter7',
        'part3/chapter8'
      ],
    },
    {
      type: 'category',
      label: 'Part 4: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'part4/chapter9',
        'part4/chapter10',
        'part4/chapter11',
        'part4/chapter12'
      ],
    },
    {
      type: 'category',
      label: 'Part 5: Humanoid Robot Development',
      items: [
        'part5/chapter13',
        'part5/chapter14',
        'part5/chapter15',
        'part5/chapter16'
      ],
    },
    {
      type: 'category',
      label: 'Part 6: Vision-Language-Action & Capstone',
      items: [
        'part6/chapter17',
        'part6/chapter18',
        'part6/chapter19',
        'part6/chapter20',
        'part6/conclusion'
      ],
    },
    {
      type: 'category',
      label: 'Project Resources',
      items: [
        'logo-showcase'
      ],
    },
  ],
};

module.exports = sidebars;