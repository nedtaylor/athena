# Configuration file for the Sphinx documentation builder.

# -- Project information
import datetime
import os
import sys
import subprocess
import shutil

def run_ford(app):
    """Run FORD to generate Fortran API documentation"""
    ford_dir = os.path.abspath(os.path.join(app.confdir, "..", ".."))
    ford_otuput = os.path.join(app.confdir, "_static", "ford")
    project_file = os.path.join(ford_dir, "ford.md")

    print(f"Running FORD with config: {project_file}")
    result = subprocess.run(["ford", project_file, "-o", ford_otuput], cwd=ford_dir, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FORD output:\n{result.stdout}")
        print(f"FORD errors:\n{result.stderr}")
    else:
        print("FORD documentation generated successfully")

from docutils.parsers.rst import roles
from docutils import nodes

def h3style_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    node = nodes.inline(text, text, classes=["h3style"])
    return [node], []

def setup(app):
    roles.register_local_role('h3style', h3style_role)
    #app.connect("builder-inited", run_ford)

# -- Project information

project = 'athena'
copyright = f'{datetime.date.today().year}, athena-developers'

# -- General configuration

# Identify the branch of the documentation
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    git_branch = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")
else:
    git_branch = "main"  # or get from git directly with subprocess

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'sphinx.ext.extlinks',
    'sphinx_copybutton'
]

extlinks = {
    'git': ('https://github.com/nedtaylor/athena/blob/' + git_branch + '/%s', 'git: %s')
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

exclude_patterns = ['_build', '.DS_Store', 'build']

# # -- Search aliases extension
# sys.path.append(os.path.abspath('_ext'))
# extensions.append('spelling_aliases')

# -- Options for HTML output

html_theme = 'furo'

templates_path = ['_templates']

# Add path for static files (will include FORD output)
html_static_path = ['_static']
html_css_files = [
    "custom.css",
]

html_theme_options = {
    # Toc options
    # 'home_page_in_toc': True,
    # 'show_toc_level': 2,  # Sidebar shows subsections down to level 2
    # 'navigation_depth': 4,
    # 'collapse_navigation': True,
    "top_of_page_buttons": [ "view", "edit" ],
    "source_repository": "https://github.com/nedtaylor/athena/",
    "source_branch": git_branch,
    "source_directory": "docs/source/",
}

html_context = {
    "display_github": True,
    "github_repo": "athena",
    "github_user": "nedtaylor",
    "github_version": git_branch,
    "conf_py_path": "docs/source/",
    "page_source_suffix": ".rst",
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
