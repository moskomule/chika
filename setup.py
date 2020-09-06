from setuptools import find_packages, setup

install_requires = ["PyYAML"]

setup(
    name="chika",
    version="0.0.1",
    description="`chika` is a dataclass-based simple and easy config tool",
    author="Ryuichiro Hataya",
    python_requires=">=3.7",
    install_requires=install_requires,
    packages=find_packages()
)
