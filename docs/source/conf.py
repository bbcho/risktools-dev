# Configuration file for the Sphinx documentation builder.
#
# Full documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

# -- Project information -----------------------------------------------------

project = "risktools"
copyright = "2022-2026, Ben Cho"
author = "Ben Cho"
release = "0.2.8.7"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "numpydoc",
]

# Autosummary settings
autosummary_generate = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# Numpydoc settings
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# Add any paths that contain templates here
templates_path = ["_templates"]

# List of patterns to ignore
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"

html_theme_options = {
    "description": "Commodity trading analytics and risk tools for Python",
    "github_user": "bbcho",
    "github_repo": "risktools-dev",
    "github_banner": True,
    "fixed_sidebar": True,
    "sidebar_collapse": True,
    "show_powered_by": False,
    "page_width": "1040px",
    "sidebar_width": "260px",
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
