# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Rambling Realms Trading Platform"
copyright = "2025, Matthew Jacobsen"
author = "Matthew Jacobsen"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.autodoc_pydantic",
]

intersphinx_mapping = {
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}


templates_path = ["_templates"]
exclude_patterns = [""]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Improve sidebar/nav behavior so items don't disappear when navigating
html_theme_options = {
    "collapse_navigation": False,  # keep siblings visible
    "sticky_navigation": True,  # keep the sidebar in view
    "navigation_depth": 4,  # show deeper nesting
}

import os
import sys
from pathlib import Path

# Ensure project root is importable for autodoc (e.g., trading.*)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
