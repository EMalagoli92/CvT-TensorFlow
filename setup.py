import json
from pathlib import Path

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("cvt_tensorflow/version.json", "r") as handle:
    version = json.load(handle)["version"]

setup(
    name="cvt_tensorflow",
    version=version,
    description="TensorFlow 2.X reimplementation of CvT: Introducing Convolutions to Vision Transformers, Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EMalagoli92/CvT-TensorFlow",
    author="EMalagoli92",
    author_email="emala.892@gmail.com",
    license="MIT",
    packages=find_packages(),
    package_data={"cvt_tensorflow": ["version.json"]},
    install_requires=Path("requirements.txt").read_text().splitlines(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    include_package_data=True,
)
