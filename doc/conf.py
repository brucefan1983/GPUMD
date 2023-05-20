#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import sphinx_rtd_theme
import subprocess

project = 'GPUMD'
author = 'The GPUMD developer team'
copyright = '2023'
site_url = 'https://gpumd.org'

version = ''
if len(version) == 0:
    process = subprocess.Popen(['git', 'tag'], stdout=subprocess.PIPE)
    stdout, _ = process.communicate()
    versions = [s for s in stdout.decode().split('\n') if len(s)]
    if len(versions) > 0:
        version = versions[-1]
    else:
        version = 'unknown'

extensions = [
    'sphinx_sitemap',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.bibtex',
    'cloud_sptheme.ext.table_styling',    
    'nbsphinx']
bibtex_bibfiles = ['publications.bib']

graphviz_output_format = 'svg'
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'
todo_include_todos = True

html_logo = "_static/logo.png"
html_favicon = "_static/logo.ico"
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
html_theme_options = {'display_version': True}
html_context = {
    'current_version': version,
    'versions':
        [('latest release',
          '{}'.format(site_url)),
         ('development version',
          '{}/dev'.format(site_url))]}
htmlhelp_basename = 'GPUMDdoc'
intersphinx_mapping = \
    {'numpy':   ('https://numpy.org/doc/stable/', None),
     'scipy':   ('https://scipy.github.io/devdocs/', None),
     'sklearn': ('https://scikit-learn.org/stable', None)}

# Settings for nbsphinx
nbsphinx_execute = 'never'

# Options for LaTeX output
_PREAMBLE = r"""
\usepackage{amsmath,amssymb}
\renewcommand{\vec}[1]{\boldsymbol{#1}}
\DeclareMathOperator*{\argmin}{\arg\!\min}
%\DeclareMathOperator{\argmin}{\arg\!\min}
"""

latex_elements = {
    'preamble': _PREAMBLE,
}
latex_documents = [
    (master_doc, 'GPUMD.tex', 'GPUMD documentation',
     'The GPUMD developer team', 'manual'),
]


# Options for manual page output
man_pages = [
    (master_doc, 'GPUMD', 'GPUMD documentation', [author], 1)
]


# Options for Texinfo output
texinfo_documents = [
    (master_doc, 'GPUMD', 'GPUMD documentation',
     author, 'GPUMD documentation', 'GPUMD', 'Miscellaneous'),
]


html_css_files = [
    'custom.css',
]
