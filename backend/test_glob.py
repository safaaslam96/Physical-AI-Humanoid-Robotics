import glob
import os

docs_dir = 'docusaurus/docs'
# Use forward slashes for the glob pattern to work cross-platform
pattern = docs_dir.replace(os.sep, '/') + '/**/*.md*'
files = glob.glob(pattern, recursive=True)

print(f'Pattern: {pattern}')
print(f'Found {len(files)} files:')
for f in files[:10]:  # Print first 10 files
    print(f'  {f}')
print(f'Total files found: {len(files)}')