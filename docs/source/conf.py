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
    "autoapi.extension",
    "sphinxcontrib.programoutput",
    "sphinx_click",
    "sphinxcontrib.typer",
    "sphinx-jsonschema",
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

# AutoAPI config
autoapi_type = "python"
autoapi_dirs = ["../../trading"]  # path to your source
autoapi_ignore = ["**/test/**"]
autoapi_keep_files = True
autoapi_member_order = "bysource"
autoapi_add_toctree_entry = True

# Global defaults for sphinx-autodoc-pydantic
# See: https://autodoc-pydantic.readthedocs.io/en/stable/users/configuration.html
# Model docs defaults
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_field_summary = False

# Field docs defaults (apply to both models and settings)
autodoc_pydantic_field_list_validators = False

# BaseSettings docs defaults (in case you document any Settings classes)
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_settings_show_config_summary = False
autodoc_pydantic_settings_show_validator_members = False
autodoc_pydantic_settings_show_validator_summary = False
autodoc_pydantic_settings_show_field_summary = False
# Render type hints in the description for better readability
autodoc_typehints = "description"

# autodoc-pydantic: show Pydantic model members (fields/validators/config) by default
autodoc_pydantic_model_members = True
# Keep member order aligned with source
autodoc_pydantic_model_member_order = "bysource"
