from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file (optional)
try:
    with open(path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

# Get the requirements from the requirements.txt file (optional).
# Comments, blank lines and direct-URL references (PEP 508 "name @ url",
# not allowed for packages published on PyPI) are filtered out: the spaCy
# model wheel stays in requirements.txt for git-based installs and is
# otherwise downloadable with `python -m spacy download en_core_web_trf`.
try:
    with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#") and "@" not in line
        ]
except FileNotFoundError:
    requirements = []

setup(
    name="teanets",
    url="https://github.com/MassimoStel/TEA_Networks.git",
    author="Sebastiano Franchini",
    author_email="franchini.sebastiano@gmail.com",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9,<3.13",
    version="0.3.1",
    license="BSD-3-Clause",
    description="Target-Event-Agent Networks: SVO extraction and analysis from text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)
