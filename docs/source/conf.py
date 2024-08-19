# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'neurokin'
copyright = '2024, E.L. Garulli, G. El Hasbani, M. Schellenberger, D. Segebarth'
author = 'E.L. Garulli, G. El Hasbani, M. Schellenberger, D. Segebarth'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', "sphinx.ext.autosummary"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_mock_imports = ["pandas", "numpy", "scipy", "tdt", "dlc2kinematics", "c3d", "yaml", "fooof", "matplotlib"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_permalinks_icon = '<span>#</span>'
html_theme = 'sphinxawesome_theme'
html_logo = "./neurokin_logo_200.png"
html_static_path = ['_static']

