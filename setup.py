from setuptools import setup, find_packages
import re

with open("./netspresso/__init__.py", "r") as f:
    content = f.read()
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)


with open("README.md", "r", encoding="UTF8") as fh:
    long_description = fh.read()

# we are using the packages in `requirements.txt` for now,
# not 100% ideal but will do
with open("requirements.txt", "r") as fh:
    install_requires = fh.read().split("\n")


setup(
    name="netspresso",
    version=version,
    author="NetsPresso",
    author_email="netspresso@nota.ai",
    description="PyNetsPresso",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nota-NetsPresso/PyNetsPresso",
    install_requires=install_requires,
    packages=find_packages(exclude=("tests",)),
    package_data={"netspresso.client": ["configs/*.ini"]},
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
