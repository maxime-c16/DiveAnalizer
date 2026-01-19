"""Setup script for DiveAnalyzer."""

from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "diveanalyzer=diveanalyzer.cli:main",
        ],
    },
)
