
from setuptools import setup, find_packages

setup(
    name="danila-grad",              # The name of your package
    version="alpha",                # The current version of your package
    packages=find_packages(exclude=["_correctness*",
                                    "tests",
                                    "_speed*",
                                    ".direnv",
                                    "architectures",
                                    "examples",
                                    ]),       # Automatically find and include your package
    install_requires=[              # Optional: List of dependencies
        "numpy==1.14.0",
        "numba==0.36.2+0.g540650d.dirty"
    ],
    author="Danila Kurganov",             # Your name
    author_email="dan.kurg@gmail.com",  # Your email
    description="Autograd in NumPy",
    long_description=open("README.md").read(),  # Optional: Long description from README
    long_description_content_type="text/markdown",  # README format
    # url="https://github.com/yourusername/my_package",  # Project URL
    classifiers=[                   # Additional package metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # python_requires='>=3.11',         # Minimum Python version required
)

