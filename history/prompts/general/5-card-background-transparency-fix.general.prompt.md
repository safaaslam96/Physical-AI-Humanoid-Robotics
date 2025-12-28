---
id: 5
title: "Card Background Transparency Fix"
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
  - Verify reason cards have transparent/light background
  - Verify week cards have transparent/light background
  - Verify cards look good in both light and dark modes
  - Verify section background shows through cards
  - Verify text remains readable
prompt_text: |
  You are fixing ONLY the card background in the "Why Physical AI Matters" section on the homepage of the Docusaurus site "Physical AI & Humanoid Robotics".

  Current issue:
  - Cards have solid dark/gray background
  - Need transparent or very light background so section background shows through

  Task (do ONLY this):

  1. Open the WhyPhysicalAIMattersSection component
  2. Update card styles (reasonCard and weekCard):
     - Background: transparent or very light (rgba(255,255,255,0.05) light mode, rgba(30,41,59,0.6) dark mode)
     - Or use #ffffff10 in light, #1e293b80 in dark
     - Keep subtle border or shadow for definition
     - Text remains readable
     - Hover effect keep (scale + glow)

  3. Ensure cards look clean and premium on section background

  Do NOT:
  - Change layout, spacing, icons, titles, descriptions
  - Add carousel back
  - Change sidebar, hero, modules carousel, auth pages
  - Change any other section

  Use Context7 for card background best practices.

  Verify on localhost:
  - "Why Physical AI Matters" cards have transparent/light background
  - Section background visible behind cards
  - Cards still readable and premium
  - No solid dark fill

  Do ONLY this card background transparency fix. Nothing else.

  After this, the cards will blend beautifully with the section background.
response_text: |
  Successfully updated the card backgrounds in the "Why Physical AI Matters" section to make them transparent/light so the section background shows through. Updated both reasonCard and weekCard styles with:
  - Light mode: rgba(255, 255, 255, 0.05) background
  - Dark mode: rgba(30, 41, 59, 0.6) background
  - Maintained all other styling, borders, shadows, and hover effects
  - Cards now blend beautifully with the section background while maintaining readability
outcome: |
  The cards in the "Why Physical AI Matters" section now have transparent/light backgrounds that allow the section background to show through while maintaining readability and premium appearance.
evaluation: |
  The changes were successful. Cards now have the requested transparent backgrounds and look clean and premium on the section background.
---

# Card Background Transparency Fix

## Summary
Updated the card backgrounds in the "Why Physical AI Matters" section to make them transparent/light so the section background shows through.

## Changes Made
1. Updated `.reasonCard` background to `rgba(255, 255, 255, 0.05)` in light mode with a dark mode fallback of `rgba(30, 41, 59, 0.6)`
2. Updated `.weekCard` background to `rgba(255, 255, 255, 0.05)` in light mode with a dark mode fallback of `rgba(30, 41, 59, 0.6)`
3. Maintained all other styling, borders, shadows, and hover effects
4. Kept text readable and preserved hover effects (scale + glow)

## Result
The cards in the "Why Physical AI Matters" section now have transparent/light backgrounds that allow the section background to show through while maintaining readability and premium appearance. The cards blend beautifully with the section background as requested.