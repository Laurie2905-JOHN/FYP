import os
import subprocess

def create_analysis_from_requirements_file(requirements_file):
    with open(requirements_file) as f:
        packages = [line.strip().split('==')[0] for line in f.readlines() if line.strip()]

    datas = [(f"C:\\Users\\lauri\\PycharmProjects\\FYP\\venv\\Lib\\site-packages\\{package.replace('-', '_')}", package) for package in packages]
    return datas


print(create_analysis_from_requirements_file('requirements.txt'))
