
from setuptools import setup, find_packages

setup(
    name="danila_grad",
    version="0.0.1",
    packages=find_packages(exclude=["_correctness*",
                                    "_extras*",
                                    "_speed*",
                                    ".direnv",
                                    "data*",
                                    "examples",
                                    "tests",
                                    ".envrc",
                                    ]),
    install_requires=[
        "numpy",
        "numba",
    ],
    author="Danila Kurganov",             # Your name
    author_email="dan.kurg@gmail.com",  # Your email
    description="Autograd in NumPy",
    long_description=open("README.md").read(),
    url="https://github.com/baubels/danila-grad",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)

