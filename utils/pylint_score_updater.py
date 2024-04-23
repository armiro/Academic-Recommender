"""
Run PyLint on all python files in the directory and add code quality score
along with appropriate badge color to the readme file icons
"""
import subprocess
import re

# run pylint and get the output
python_file_names = subprocess.check_output(['git', 'ls-files', '*.py']).decode().splitlines()
pylint_output = subprocess.check_output(['pylint', '--exit-zero'] + python_file_names)

# extract the pylint score using regex
score_match = re.search(r'Your code has been rated at (\d+.\d+)/10', pylint_output.decode())
pylint_score = float(score_match.group(1)) if score_match else 0.0

SCORE_COLOR = 'brightgreen' if pylint_score >= 8.0 else 'yellow' if pylint_score >= 6.0 else 'red'

# update readme.md with the pylint badge and score
with open('README.md', mode='r', encoding='utf-8') as readme_file:
    readme_content = readme_file.read()

with open('README.md', mode='w', encoding='utf-8') as readme_file:
    updated_content = re.sub(pattern=r"!\[Pylint\].*",
                             repl=f"![Pylint](https://img.shields.io/badge/PyLint-"
                                  f"{pylint_score}/10-{SCORE_COLOR})",
                             string=readme_content)
    readme_file.write(updated_content)
