---
sidebar_position: 5
---

# اپنی سائٹ کو اسٹیج کریں

Docusaurus سائٹ کو اسٹیج کرنا آسان ہے!

## ایک بنیادی بنانا

سائٹ کو بنانے کے لیے، مندرجہ ذیل کمانڈ چلائیں:

```bash
npm run build
```

یہ بنیادی فائلیں `build` ڈائرکٹری میں تخلیق کرے گا۔

## اسٹیج کے انتخابات

### GitHub پیجز کے لیے

1. `docusaurus.config.js` میں اسٹیج کے ترتیبات کو اپ ڈیٹ کریں:

   ```js title="docusaurus.config.js"
   export default {
     // ...
     url: 'https://your-github-username.github.io',
     baseUrl: '/your-project-name/',
     projectName: 'your-project-name',
     organizationName: 'your-github-username',
     // ...
   };
   ```

2. GitHub پر ایک ریپوزٹری تخلیق کریں

3. `gh-pages` برانچ پر اپنی سائٹ کو اسٹیج کرنے کے لیے مندرجہ ذیل کمانڈ چلائیں:

   ```bash
   GIT_USER=<your-github-username> yarn deploy
   ```

### Netlify کے لیے

1. `docusaurus.config.js` میں اسٹیج کے ترتیبات کو اپ ڈیٹ کریں:

   ```js title="docusaurus.config.js"
   export default {
     // ...
     url: 'https://your-netlify-subdomain.netlify.app',
     baseUrl: '/',
     // ...
   };
   ```

2. اپنی `build` فولڈر کو Netlify کے ساتھ کنیکٹ کریں

## اگلے اقدامات

- [اپنی سائٹ کو کسٹمائز کریں](./customize-your-site.md) کے بارے میں مزید جانیں
- [زیادہ تفصیل کے لیے دستاویزات دیکھیں](https://docusaurus.io/docs/deployment)