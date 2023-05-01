# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['dash_app.py'],
    pathex=[],
    binaries=[],
datas=[('venv\Lib\site-packages\altgraph', 'altgraph'), ('venv\Lib\site-packages\ansi2html', 'ansi2html'), ('venv\Lib\site-packages\asttokens', 'asttokens'), ('venv\Lib\site-packages\async-generator', 'async-generator'), ('venv\Lib\site-packages\attrs', 'attrs'), ('venv\Lib\site-packages\backcall', 'backcall'), ('venv\Lib\site-packages\blinker', 'blinker'), ('venv\Lib\site-packages\cachelib', 'cachelib'), ('venv\Lib\site-packages\certifi', 'certifi'), ('venv\Lib\site-packages\cffi', 'cffi'), ('venv\Lib\site-packages\charset-normalizer', 'charset-normalizer'), ('venv\Lib\site-packages\click', 'click'), ('venv\Lib\site-packages\colorama', 'colorama'), ('venv\Lib\site-packages\comm', 'comm'), ('venv\Lib\site-packages\contourpy', 'contourpy'), ('venv\Lib\site-packages\cycler', 'cycler'), ('venv\Lib\site-packages\dash', 'dash'), ('venv\Lib\site-packages\dash-bootstrap-components', 'dash-bootstrap-components'), ('venv\Lib\site-packages\dash-core-components', 'dash-core-components'), ('venv\Lib\site-packages\dash-extensions', 'dash-extensions'), ('venv\Lib\site-packages\dash-html-components', 'dash-html-components'), ('venv\Lib\site-packages\dash-mantine-components', 'dash-mantine-components'), ('venv\Lib\site-packages\dash-table', 'dash-table'), ('venv\Lib\site-packages\dash-uploader', 'dash-uploader'), ('venv\Lib\site-packages\debugpy', 'debugpy'), ('venv\Lib\site-packages\decorator', 'decorator'), ('venv\Lib\site-packages\dill', 'dill'), ('venv\Lib\site-packages\et-xmlfile', 'et-xmlfile'), ('venv\Lib\site-packages\exceptiongroup', 'exceptiongroup'), ('venv\Lib\site-packages\executing', 'executing'), ('venv\Lib\site-packages\Flask', 'Flask'), ('venv\Lib\site-packages\fonttools', 'fonttools'), ('venv\Lib\site-packages\h11', 'h11'), ('venv\Lib\site-packages\idna', 'idna'), ('venv\Lib\site-packages\install', 'install'), ('venv\Lib\site-packages\ipykernel', 'ipykernel'), ('venv\Lib\site-packages\ipython', 'ipython'), ('venv\Lib\site-packages\itsdangerous', 'itsdangerous'), ('venv\Lib\site-packages\jedi', 'jedi'), ('venv\Lib\site-packages\Jinja2', 'Jinja2'), ('venv\Lib\site-packages\jupyter-dash', 'jupyter-dash'), ('venv\Lib\site-packages\jupyter_client', 'jupyter_client'), ('venv\Lib\site-packages\jupyter_core', 'jupyter_core'), ('venv\Lib\site-packages\kiwisolver', 'kiwisolver'), ('venv\Lib\site-packages\MarkupSafe', 'MarkupSafe'), ('venv\Lib\site-packages\more-itertools', 'more-itertools'), ('venv\Lib\site-packages\multiprocess', 'multiprocess'), ('venv\Lib\site-packages\nest-asyncio', 'nest-asyncio'), ('venv\Lib\site-packages\numpy', 'numpy'), ('venv\Lib\site-packages\openpyxl', 'openpyxl'), ('venv\Lib\site-packages\outcome', 'outcome'), ('venv\Lib\site-packages\packaging', 'packaging'), ('venv\Lib\site-packages\pandas', 'pandas'), ('venv\Lib\site-packages\parso', 'parso'), ('venv\Lib\site-packages\patsy', 'patsy'), ('venv\Lib\site-packages\pefile', 'pefile'), ('venv\Lib\site-packages\pickleshare', 'pickleshare'), ('venv\Lib\site-packages\Pillow', 'Pillow'), ('venv\Lib\site-packages\platformdirs', 'platformdirs'), ('venv\Lib\site-packages\plotly', 'plotly'), ('venv\Lib\site-packages\plotly-express', 'plotly-express'), ('venv\Lib\site-packages\plotly-resampler', 'plotly-resampler'), ('venv\Lib\site-packages\prompt-toolkit', 'prompt-toolkit'), ('venv\Lib\site-packages\psutil', 'psutil'), ('venv\Lib\site-packages\pure-eval', 'pure-eval'), ('venv\Lib\site-packages\pycparser', 'pycparser'), ('venv\Lib\site-packages\Pygments', 'Pygments'), ('venv\Lib\site-packages\pyinstaller', 'pyinstaller'), ('venv\Lib\site-packages\pyinstaller-hooks-contrib', 'pyinstaller-hooks-contrib'), ('venv\Lib\site-packages\Pympler', 'Pympler'), ('venv\Lib\site-packages\pyparsing', 'pyparsing'), ('venv\Lib\site-packages\PySocks', 'PySocks'), ('venv\Lib\site-packages\python-dateutil', 'python-dateutil'), ('venv\Lib\site-packages\pytz', 'pytz'), ('venv\Lib\site-packages\pywin32', 'pywin32'), ('venv\Lib\site-packages\pywin32-ctypes', 'pywin32-ctypes'), ('venv\Lib\site-packages\pyzmq', 'pyzmq'), ('venv\Lib\site-packages\requests', 'requests'), ('venv\Lib\site-packages\retrying', 'retrying'), ('venv\Lib\site-packages\scipy', 'scipy'), ('venv\Lib\site-packages\six', 'six'), ('venv\Lib\site-packages\sniffio', 'sniffio'), ('venv\Lib\site-packages\sortedcontainers', 'sortedcontainers'), ('venv\Lib\site-packages\stack-data', 'stack-data'), ('venv\Lib\site-packages\statsmodels', 'statsmodels'), ('venv\Lib\site-packages\tenacity', 'tenacity'), ('venv\Lib\site-packages\tornado', 'tornado'), ('venv\Lib\site-packages\trace-updater', 'trace-updater'), ('venv\Lib\site-packages\traitlets', 'traitlets'), ('venv\Lib\site-packages\trio', 'trio'), ('venv\Lib\site-packages\trio-websocket', 'trio-websocket'), ('venv\Lib\site-packages\tsdownsample', 'tsdownsample'), ('venv\Lib\site-packages\urllib3', 'urllib3'), ('venv\Lib\site-packages\wcwidth', 'wcwidth'), ('venv\Lib\site-packages\Werkzeug', 'Werkzeug'), ('venv\Lib\site-packages\wsproto', 'wsproto')],
    hiddenimports=[],
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
    name='dash_app',
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
