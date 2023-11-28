from setuptools import setup

setup(
    name="datan",
    version="1.0",
    author="Finley Gibson",
    author_email="f.j.gibson@exeter.ac.uk",
    packages = ["datan"],
    package_dir = {"": "src"},
    install_requires=[
        "numpy",
        "tqdm",
        "toml",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ],
)


