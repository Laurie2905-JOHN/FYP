import os
import subprocess

def main():
    requirements_file = 'requirements.txt'

    with open(requirements_file, 'r') as f:
        lines = f.readlines()

    packages = [line.strip().split('==')[0] for line in lines if line.strip()]

    hidden_imports = ' '.join([f'--hidden-import={package}' for package in packages])

    pyinstaller_command = f'pyinstaller --onefile {hidden_imports} dash_app.py'

    print(f'Running: {pyinstaller_command}')
    subprocess.run(pyinstaller_command, shell=True)

if __name__ == '__main__':
    main()