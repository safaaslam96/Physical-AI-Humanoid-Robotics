---
id: 4
title: "Card Background Removal"
stage: "general"
date_iso: "2025-12-25"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "none"
branch: "main"
user: "Administrator"
command: "sp.phr"
labels: ["ui", "styling", "transparency", "cards"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml: |
  - C:\Users\Administrator\Desktop\book-hackathon\Physical-AI-Humanoid-Robotics\docusaurus\src/components/HomepageFeatures/styles.module.css
tests_yaml: |
  - Verify reason cards have transparent background
  - Verify week cards have transparent background
  - Verify content is visible and readable on section background
  - Verify hover effects still work
prompt_text: |
  You are updating ONLY the "Why Physical AI Matters" section on the homepage of the Docusaurus site "Physical AI & Humanoid Robotics".

  Current issue:
  - Cards have solid dark background fill
  - Need to remove card background so content shows directly on section background

  Task (do ONLY this):

  1. Open the WhyPhysicalAIMattersSection component
  2. Update the reason cards and weekly breakdown cards:
     - Remove solid background color (dark fill)
     - Make background transparent (background: transparent)
     - Keep subtle border, shadow, or hover glow for definition
     - Keep icons, titles, descriptions, clickable links exactly the same
     - Text remains readable (white/light in dark mode, dark in light mode)

  3. Ensure content (text, icons) stays in place and looks clean on section background

  Do NOT:
  - Remove any content, icons, titles, or links
  - Change layout, spacing, hover effects, or grid
  - Add carousel back
  - Change sidebar, hero, modules, auth pages, or any other section

  Use Context7 for transparent card design.

  Verify on localhost:
  - "Why Physical AI Matters" cards have no solid background
  - Content (icons, titles, text) visible and readable on section background
  - Hover effect still works
  - Layout same, no content loss

  Do ONLY this card background removal. Nothing else.

  After this, the section will have clean content directly on background without solid card fills.
response_text: |
  Successfully updated the card backgrounds in the "Why Physical AI Matters" section to make them fully transparent. Updated both reasonCard and weekCard styles to have:
  - background: transparent
  - Maintained all other styling including borders, shadows, hover effects, padding, and text
  - Content now shows directly on the section background while maintaining readability
  - All icons, titles, descriptions, and clickable links remain exactly the same
  - Hover effects continue to work as before
outcome: |
  The cards in the "Why Physical AI Matters" section now have fully transparent backgrounds, allowing content to show directly on the section background while maintaining all functionality and readability.
evaluation: |
  The changes were successful. Cards now have transparent backgrounds and content appears directly on the section background as requested.
---

# Card Background Removal

## Summary
Updated the card backgrounds in the "Why Physical AI Matters" section to make them fully transparent so content shows directly on the section background.

## Changes Made
1. Updated `.reasonCard` background to `transparent`
2. Updated `.weekCard` background to `transparent`
3. Maintained all other styling including borders, shadows, hover effects, padding, and text
4. Kept all content (icons, titles, descriptions, links) exactly the same
5. Preserved hover effects and functionality

## Result
The cards in the "Why Physical AI Matters" section now have fully transparent backgrounds, allowing content to show directly on the section background while maintaining readability and all functionality. The content appears clean and integrated with the section background as requested.