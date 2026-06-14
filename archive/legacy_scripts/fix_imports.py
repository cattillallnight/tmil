import os

def replace_in_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
        
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

replacements = {
    "from src.utils import": "from src.utils import",
    "import src.utils as utils": "import src.utils as utils",
    "src.model": "src.model",
    "src.train": "src.train",
    "src.evaluate": "src.evaluate",
    "src.feature_extraction": "src.feature_extraction"
}

# Fix src and scripts
for d in ['src', 'scripts', 'scratch', 'ground_truth']:
    for root, _, files in os.walk(d):
        for file in files:
            if file.endswith('.py'):
                replace_in_file(os.path.join(root, file), replacements)

print("Imports updated!")
