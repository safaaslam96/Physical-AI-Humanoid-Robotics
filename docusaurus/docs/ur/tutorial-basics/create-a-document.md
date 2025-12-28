---
sidebar_position: 2
---

# ایک دستاویز تخلیق کریں

دستاویزات صفحات کے **گروہ** ہیں جو مندرجہ ذیل کے ذریعے منسلک ہیں:

- ایک **سائڈ بار**
- **پچھلا/اگلا نیویگیشن**
- **ورژننگ**

## اپنی پہلی دستاویز تخلیق کریں

`docs/hello.md` میں ایک مارک ڈاؤن فائل تخلیق کریں:

```md title="docs/hello.md"
# ہیلو

یہ میرا **پہلا ڈوکوسورس دستاویز** ہے!
```

یہ ایک نیا صفحہ تخلیق کرے گا جس کا URL `/docs/hello` ہو گا۔

## سائڈ بار شامل کریں

آپ اپنی دستاویز کو ایک سائڈ بار میں شامل کر سکتے ہیں۔

`sidebars.js` میں اپنی دستاویز کا اندراج شامل کریں:

```js title="sidebars.js"
export default {
  tutorialSidebar: [
    'intro',
    // highlight-next-line
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
};
```