#!/usr/bin/env python3
"""
Pack botzone submission files (code only, model goes to user storage)
Usage: python pack_for_botzone.py

IMPORTANT: 
- This script creates a zip file with ONLY code files at the ROOT level.
- Model file (.pkl) should be uploaded via Botzone "User Storage" to data/ directory.
- In Botzone, the model will be accessible at: data/model.pkl
"""
import os
import zipfile

# Configuration
SOURCE_DIR = '.'
OUTPUT_ZIP = 'botzone-submit.zip'

def main():
    # Clean old output
    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)
    
    # Files to include (at root level of zip)
    files_to_include = ['__main__.py', 'model.py', 'feature.py', 'agent.py']
    
    # Create zip with files at root level
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_include:
            src = os.path.join(SOURCE_DIR, f)
            if os.path.exists(src):
                zf.write(src, f)
                print(f'Added to zip: {f}')
            else:
                print(f'WARNING: {f} not found!')
    
    print(f'\n✅ Botzone code submission ready: {OUTPUT_ZIP}')
    print('\n📋 Next steps:')
    print('   1. Upload this zip to Botzone as your bot code')
    print('   2. Rename your model file to "model.pkl"')
    print('   3. Upload "model.pkl" via Botzone User Storage to data/ directory')
    
    print('\n📦 Zip contents:')
    with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zf:
        for name in zf.namelist():
            print(f'   {name}')

if __name__ == '__main__':
    main()
