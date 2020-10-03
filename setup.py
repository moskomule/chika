from setuptools import find_packages, setup

install_requires = ["PyYAML"]

setup(
    name="chika",
    version="0.0.2",
    description="chika: a dataclass-based simple and easy config tool",
    author="Ryuichiro Hataya",
    python_requires=">=3.8",
    install_requires=install_requires,
    packages=find_packages(exclude=("tests",))
)
