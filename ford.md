project:
summary: A Fortran library for implementing neural networks
src_dir: ./src
output_dir: docs/html
output: html
preprocess: false
predocmark: !!
fpp_extensions: f90
                F90
display: public
         protected
         private
source: true
graph: true
search: true
md_extensions: markdown.extensions.toc
coloured_edges: true
sort: permission-alpha
author: Ned Thaddeus Taylor
github: https://github.com/nedtaylor
print_creation_date: true
creation_date: %Y-%m-%d %H:%M %z
project_github: https://github.com/nedtaylor/athena
project_download: https://github.com/nedtaylor/athena/releases
github: https://github.com/nedtaylor
externalize: True
external: diffstruc: https://diffstruc.readthedocs.io/en/latest/_static/ford/
            graphstruc: https://graphstruc.readthedocs.io/en/latest/_static/ford/

{!README.md!}
