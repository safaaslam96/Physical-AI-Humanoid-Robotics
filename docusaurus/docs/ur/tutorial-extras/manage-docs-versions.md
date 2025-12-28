---
sidebar_position: 1
---

# دستاویزات کے ورژن کا نظم کریں

ڈوکوسورس آپ کی دستاویزات کے متعدد ورژن کا نظم کر سکتا ہے۔

## ایک دستاویزات کا ورژن بنائیں

اپنے پروجیکٹ کا ورژن 1.0 جاری کریں:

```bash
npm run docusaurus docs:version 1.0
```

`docs` فولڈر کو `versioned_docs/version-1.0` میں کاپی کر دیا جاتا ہے اور `versions.json` تخلیق ہو جاتا ہے۔

آپ کی دستاویزات اب 2 ورژن رکھتی ہیں:

- `1.0` در `http://localhost:3000/docs/` ورژن 1.0 دستاویزات کے لیے
- `current` در `http://localhost:3000/docs/next/` **آئندہ، غیر جاری دستاویزات** کے لیے

## ایک ورژن ڈراپ ڈاؤن شامل کریں

ورژن کے درمیان بے رکاوٹ نیویگیشن کے لیے، ایک ورژن ڈراپ ڈاؤن شامل کریں۔

`docusaurus.config.js` فائل میں ترمیم کریں:

```js title="docusaurus.config.js"
export default {
  themeConfig: {
    navbar: {
      items: [
        // highlight-start
        {
          type: 'docsVersionDropdown',
        },
        // highlight-end
      ],
    },
  },
};
```

دستاویزات کا ورژن ڈراپ ڈاؤن آپ کے نیوی گیشن بار میں ظاہر ہوتا ہے:

## موجودہ ورژن کو اپ ڈیٹ کریں

یہ ممکن ہے کہ ورژن والی دستاویزات کو ان کے متعلقہ فولڈر میں ترمیم کیا جا سکے:

- `versioned_docs/version-1.0/hello.md` اپ ڈیٹ کرتا ہے `http://localhost:3000/docs/hello`
- `docs/hello.md` اپ ڈیٹ کرتا ہے `http://localhost:3000/docs/next/hello`