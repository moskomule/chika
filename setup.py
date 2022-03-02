from setuptools import find_packages, setup

setup(
    name="chika",
    version="0.1.0",
    description="chika: a dataclass-based simple and easy config tool",
    author="Ryuichiro Hataya",
    url="https://github.com/moskomule/chika",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    install_requires=["pyyaml"],
    packages=find_packages(exclude=("tests",)),
    long_description=open("README.md", mode="r").read(),
    long_description_content_type="text/markdown",
)
