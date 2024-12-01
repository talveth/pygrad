
from setuptools import find_packages, setup

setup(
    name="pygradproject",
    version="0.0.2",
    packages=find_packages(exclude=[
                                    "examples",
                                    "tests",
                                    "docs",
                                    ]),
    install_requires=[
        "numba>=0.60.0",
        "numpy<=2.0.2",
        "opt_einsum>=3.4.0",
    ],
    author="Danila Kurganov",
    author_email="dan.kurg@gmail.com",
    description="Lightweight Automatic Differentiation in Python.",
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    url="https://github.com/baubels/pygrad",
    project_urls={
        "Documentation": "https://baubels.github.io/pygrad/",
        "Source Code": "https://github.com/baubels/pygrad",
        "Issue Tracker": "https://github.com/baubels/pygrad/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    extras_require={
        "examples": ["dill", "tqdm"],
        "dev": ["pytest", "sphinx", "torch", "numpy==1.23.0"]
    },
)
