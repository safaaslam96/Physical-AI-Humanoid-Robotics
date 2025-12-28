---
sidebar_position: 2
---

# اپنی سائٹ کا ترجمہ کریں

چلو `docs/intro.md` کا فرانسیسی میں ترجمہ کریں۔

## i18n کی ترتیب

`docusaurus.config.js` کو فرانسیسی لوکلائزیشن کے لیے حمایت شامل کرنے کے لیے تبدیل کریں:

```js title="docusaurus.config.js"
export default {
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
  },
};
```

## ایک دستاویز کا ترجمہ

`docs/intro.md` فائل کو `i18n/ur` فولڈر میں کاپی کریں:

```bash
mkdir -p i18n/ur/docusaurus-plugin-content-docs/current/

cp docs/intro.md i18n/ur/docusaurus-plugin-content-docs/current/intro.md
```

`i18n/ur/docusaurus-plugin-content-docs/current/intro.md` کا اردو میں ترجمہ کریں۔

## اپنی مقامی سائٹ شروع کریں

اردو لوکلائزیشن پر اپنی سائٹ شروع کریں:

```bash
npm run start -- --locale ur
```

آپ کی مقامی سائٹ [http://localhost:3000/ur/](http://localhost:3000/ur/) پر دستیاب ہے اور 'شروع کریں' صفحہ ترجمہ شدہ ہے۔

:::caution

ڈیولپمنٹ میں، آپ ایک وقت میں صرف ایک لوکلائزیشن استعمال کر سکتے ہیں۔

:::

## ایک لوکل ڈراپ ڈاؤن شامل کریں

زبانوں کے درمیان بے رکاوٹ نیویگیشن کے لیے، ایک لوکل ڈراپ ڈاؤن شامل کریں۔

`docusaurus.config.js` فائل میں تبدیلی کریں:

```js title="docusaurus.config.js"
export default {
  themeConfig: {
    navbar: {
      items: [
        // highlight-start
        {
          type: 'localeDropdown',
        },
        // highlight-end
      ],
    },
  },
};
```

لوکل ڈراپ ڈاؤن اب آپ کے نیویگیشن بار میں ظاہر ہوتا ہے۔

## اپنی مقامی سائٹ بنائیں

ایک مخصوص لوکلائزیشن کے لیے اپنی سائٹ بنائیں:

```bash
npm run build -- --locale ur
```

یا ایک بار میں تمام لوکلائزیشنز شامل کرکے اپنی سائٹ بنائیں:

```bash
npm run build
```