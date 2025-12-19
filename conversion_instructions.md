# Instructions for Converting HTML Book Cover to Image

## Method 1: Using Browser Developer Tools (Recommended)

1. Open the generated HTML file in Chrome or Edge:
   - Right-click on the HTML file → "Open with" → Chrome or Edge
   - Or drag and drop the file into the browser window

2. Press F12 to open Developer Tools
3. Press Ctrl+Shift+P (Windows/Linux) or Cmd+Shift+P (Mac) to open Command Menu
4. Type "Capture full size screenshot" and press Enter
5. The browser will save a high-resolution PNG of the entire page

## Method 2: Using Online HTML to Image Converters

Several online services can convert HTML to images:
- htmlcsstoimage.com
- convertio.co/html-to-png/
- cloudconvert.com/html-to-png

## Method 3: Using Puppeteer (Node.js)

If you have Node.js installed:

```bash
npm install puppeteer
```

Create a script:

```javascript
const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 2400 });
  await page.goto('file:///path/to/your/book_cover.html', {
    waitUntil: 'networkidle2'
  });
  await page.screenshot({
    path: 'book_cover.png',
    fullPage: true
  });
  await browser.close();
})();
```

## Method 4: Using Adobe Photoshop or GIMP

1. Open the HTML file in a browser
2. Take a screenshot and open in your preferred image editor
3. Crop to the exact dimensions: 1600x2400px
4. Save as PNG or JPEG with high quality

## Quality Tips

- The HTML is designed for 1600x2400px resolution
- Ensure you capture the full height without scrolling artifacts
- Save in PNG format for lossless compression and transparency support
- For print use, consider saving at higher DPI if needed
- The design includes animations that will be captured in the static image

## Final Output

Your book cover will feature:
- Modern dark blue gradient background with purple accents
- Glowing "Physical AI and Humanoid Robotics" title
- Wireframe humanoid robot with circuit patterns
- Professional, tech-forward aesthetic
- Suitable for both digital and print applications