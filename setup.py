import setuptools
import serieset

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [line.strip() for line in open("requirements.txt").readlines()]

setuptools.setup(
    name="serieset",
    version=serieset.__version__,
    author="liaoyuhua",
    author_email="ml.liaoyuhua@gmail.com",
    description="Easy Time Series Dataset with PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liaoyuhua/serieset",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)