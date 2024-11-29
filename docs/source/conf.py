
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

base_directory = 'autograd/'

# Get all subdirectories in the base directory
for root, dirs, files in os.walk(base_directory):
    for dir_name in dirs:
        subdir_path = os.path.join(root, dir_name)
        # Insert the subdirectory path into sys.path
        if subdir_path not in sys.path:
            sys.path.insert(0, subdir_path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'danila-grad'
copyright = '2024, Danila Kurganov'
author = 'Danila Kurganov'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
]
autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'about.html',
        'searchfield.html',
        'navigation.html',
        'relations.html',
    ]
}

html_theme_options = {
    "description": "A light autodifferentiation engine written in Python, based on NumPy and Numba.",
    "github_user": "baubels",
    "github_repo": "danila-grad",
    "fixed_sidebar": True,
    "github_banner": True,
}
