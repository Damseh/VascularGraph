=========
VascGraph
=========

.. image:: https://img.shields.io/pypi/v/VascGraph.svg
        :target: https://pypi.python.org/pypi/VascGraph

.. image:: https://img.shields.io/travis/Damseh/VascGraph.svg
        :target: https://travis-ci.com/Damseh/VascGraph

.. image:: https://readthedocs.org/projects/VascGraph/badge/?version=latest
        :target: https://VascGraph.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

A Python package to generate graph-based anatomical models of vascular structures. 

Papers
------
@article{damseh2019laplacian,
  title={Laplacian Flow Dynamics on Geometric Graphs for Anatomical Modeling of Cerebrovascular Networks},
  author={Damseh, Rafat and Delafontaine-Martel, Patrick and Pouliot, Philippe and Cheriet, Farida and Lesage, Frederic},
  journal={arXiv preprint arXiv:1912.10003},
  year={2019}}

@article{damseh2018automatic,
  title={Automatic Graph-Based Modeling of Brain Microvessels Captured With Two-Photon Microscopy},
  author={Damseh, Rafat and Pouliot, Philippe and Gagnon, Louis and Sakadzic, Sava and Boas, David and Cheriet, Farida and Lesage, Frederic},
  journal={IEEE journal of biomedical and health informatics},
  volume={23},
  number={6},
  pages={2551--2562},
  year={2018},
  publisher={IEEE}}


To install
----------

conda create -n test python=3.7 spyder matplotlib scipy networkx=2.2 mayavi

source activate test

python setup.py install

To test
-------

python -i demo.py


* Free software: MIT license
* Documentation: https://VascGraph.readthedocs.io.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
