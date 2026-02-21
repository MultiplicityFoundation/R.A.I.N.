# -*- mode: python ; coding: utf-8 -*-
import sys
import os

block_cipher = None

# Include all .md and .txt files in the library
datas = []
for file in os.listdir('.'):
    if file.endswith('.md') or file.endswith('.txt'):
        datas.append((file, '.'))

# Add james_library folder if needed (current dir is likely usually sufficient if run from there)
# But let's be safe and include subfolders too
datas.append(('rlm-main', 'rlm-main'))

a = Analysis(
    ['rain_lab.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'hello_os', 
        'rlm', 
        'scipy', 
        'numpy', 
        'matplotlib', 
        'scipy.special.cython_special',
        'sklearn.utils._typedefs',
        'scipy.integrate.vode',
        'scipy.integrate.lsoda'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='rain_lab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
