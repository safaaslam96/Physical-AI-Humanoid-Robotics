import glob
import os

docs_dir = 'docusaurus/docs'

# Try different patterns
patterns = [
    'docusaurus/docs/**/*.md',
    'docusaurus/docs/**/*.mdx',
    'docusaurus/docs/**/*.md*',
    'docusaurus/docs/**/*.*md*',
    'docusaurus/docs/*.*',
    'docusaurus/docs/**/*'
]

for pattern in patterns:
    files = glob.glob(pattern, recursive=True)
    print(f'Pattern: {pattern} -> Found {len(files)} files')
    if len(files) > 0:
        for f in files[:3]:  # Print first 3 files
            print(f'  {f}')
        print()