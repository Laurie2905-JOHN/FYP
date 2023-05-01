import os
import subprocess


def main():
    requirements_file = r'requirements.txt'
    main_script = r'dash_app.py'

    # Read packages from requirements.txt
    with open(requirements_file, 'r') as f:
        lines = f.readlines()
    packages = [line.strip().split('==')[0] for line in lines if line.strip()]

    # Create a spec file
    subprocess.run(f'pyi-makespec --onefile {main_script}', shell=True)

    spec_file = main_script.replace('.py', '.spec')

    # Read spec file
    with open(spec_file, 'r') as f:
        spec_content = f.readlines()

    # Find the 'datas' line in the spec file
    for i, line in enumerate(spec_content):
        if 'datas=' in line:
            break

    # Prepare the datas list for packages
    venv_path = os.path.join('venv', 'Lib', 'site-packages')
    datas = [f"('{os.path.join(venv_path, package)}', '{package}')" for package in packages]
    datas_line = f"datas=[{', '.join(datas)}],\n"

    # Update the 'datas' line in the spec file
    spec_content[i] = datas_line

    # Write the updated spec file
    with open(spec_file, 'w') as f:
        f.writelines(spec_content)

    # Run PyInstaller with the updated spec file
    subprocess.run(f'pyinstaller {spec_file}', shell=True)


if __name__ == '__main__':
    main()