---
sidebar_position: 1
---

# ایک صفحہ تخلیق کریں

## ایک صفحہ تخلیق کریں

Docusaurus کے ساتھ، **React** کمپوننٹس کے ذریعے صفحات تخلیق کیے جاتے ہیں۔

`src/pages/` میں ایک فائل تخلیق کریں:

```jsx title="src/pages/my-react-page.js"
import React from 'react';
import Layout from '@theme/Layout';

export default function MyReactPage() {
  return (
    <Layout title="My React page">
      <div className="container margin-vert--xl">
        <div className="row">
          <div className="col col--6 col--offset-3">
            <h1>میرا صفحہ</h1>
            <p>یہ میرا <strong>React</strong> صفحہ ہے</p>
          </div>
        </div>
      </div>
    </Layout>
  );
}
```

یہ ایک نیا صفحہ تخلیق کرے گا جس کا URL `/my-react-page` ہو گا۔

## اگلے اقدامات

- [ایک دستاویز تخلیق کریں](./create-a-document.md) کے بارے میں مزید جانیں
- [اپنے سائٹ کو کسٹمائز کریں](./customize-your-site.md) کے بارے میں مزید جانیں