import os
import glob

docs_dir = "../docusaurus/docs"  # Relative to backend directory
print(f"Current working directory: {os.getcwd()}")
print(f"Docs directory exists: {os.path.exists(docs_dir)}")

# Test the original pattern
pattern = os.path.join(docs_dir, "**", "*.md*")
print(f"Pattern: {pattern}")
print(f"Pattern with forward slashes: {pattern.replace(os.sep, '/')}")

# Try with glob
files = glob.glob(pattern, recursive=True)
print(f"Files found with original pattern: {len(files)}")

# Try with forward slashes
pattern_fs = pattern.replace(os.sep, "/")
files_fs = glob.glob(pattern_fs, recursive=True)
print(f"Files found with forward slashes: {len(files_fs)}")

# Try a simple pattern
simple_pattern = "../docusaurus/docs/**/*.md"
files_simple = glob.glob(simple_pattern, recursive=True)
print(f"Files found with simple pattern: {len(files_simple)}")

# List all files to see the structure
all_files = glob.glob("../docusaurus/docs/**/*", recursive=True)
md_files = [f for f in all_files if f.endswith(('.md', '.mdx'))]
print(f"Total markdown files found: {len(md_files)}")
for f in md_files[:5]:
    print(f"  {f}")