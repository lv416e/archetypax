"""Script to build documentation automatically.

This script will:
1. Generate API documentation from docstrings
2. Build HTML documentation using Sphinx.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command):
    """Run a shell command and print output."""
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for line in iter(process.stdout.readline, b""):
        sys.stdout.write(line.decode("utf-8"))

    process.wait()
    if process.returncode != 0:
        print(f"Command failed with exit code {process.returncode}")
        sys.exit(process.returncode)


def main():
    """Build the documentation."""
    # Get the docs directory (where this script is)
    docs_dir = Path(__file__).parent.absolute()
    api_dir = docs_dir / "reference"

    # Create api directory if it doesn't exist
    api_dir.mkdir(exist_ok=True)

    # Clean existing .rst files in the api directory
    for f in api_dir.glob("*.rst"):
        f.unlink()

    print("Generating API documentation...")
    run_command(f"sphinx-apidoc -f -o {api_dir} ../archetypax")

    print("Building HTML documentation...")
    run_command(f"sphinx-build -b html {docs_dir} {docs_dir}/_build/html")

    print("Documentation built successfully!")
    print(f"Open {docs_dir}/_build/html/index.html in your browser to view it.")


if __name__ == "__main__":
    main()
