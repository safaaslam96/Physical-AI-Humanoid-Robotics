---
sidebar_position: 4
---

# اپنی سائٹ کو کسٹمائز کریں

## کلر پلیٹ کو تبدیل کریں

`src/css/custom.css` کو کھولیں اور ایک کسٹم کلر شامل کریں:

```css title="src/css/custom.css"
:root {
  --ifm-color-primary: #29784c;
  --ifm-color-primary-dark: #277148;
  --ifm-color-primary-darker: #256a45;
  --ifm-color-primary-darkest: #215732;
  --ifm-color-primary-light: #2b8250;
  --ifm-color-primary-lighter: #32925f;
  --ifm-color-primary-lightest: #4da27b;
}
```

## کمپوننٹ کو کسٹمائز کریں

### ہیڈر کو کسٹمائز کریں

`src/components/Header/index.js` میں ایک کمپوننٹ تخلیق کریں:

```jsx title="src/components/Header/index.js"
import React from 'react';

export default function Header() {
  return (
    <header>
      <h1>میری کسٹم ہیڈر</h1>
    </header>
  );
}
```

### فوٹر کو کسٹمائز کریں

`docusaurus.config.js` میں فوٹر کو کسٹمائز کریں:

```js title="docusaurus.config.js"
export default {
  // ...
  footer: {
    style: 'dark',
    links: [
      {
        title: 'Docs',
        items: [
          {
            label: 'Tutorial',
            to: '/docs/tutorial/intro',
          },
        ],
      },
      {
        title: 'Community',
        items: [
          {
            label: 'Stack Overflow',
            href: 'https://stackoverflow.com/questions/tagged/docusaurus',
          },
        ],
      },
      {
        title: 'More',
        items: [
          {
            label: 'Blog',
            to: '/blog',
          },
        ],
      },
    ],
    copyright: `کاپی رائٹ © ${new Date().getFullYear()} میرا پروجیکٹ، Inc. تیار کردہ Docusaurus.`,
  },
};
```

## اگلے اقدامات

- [اپنی سائٹ کو اسٹیج کریں](./deploy-your-site.md) کے بارے میں مزید جانیں
- [ترجمہ کے سائٹ کے بارے میں مزید جانیں](../tutorial-extras/translate-your-site.md)