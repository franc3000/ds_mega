#!/usr/bin/env python

from setuptools import setup, find_packages

version = '0.0.1'

setup(
    name='ds_mega',
    version=version,
    description="ds_mega",
    author="Audantic",
    author_email='franklin.sarkett@gmail.com',
    url='https://github.com/franc3000/pareto',
    packages=find_packages(),
    package_dir={'ds_mega': 'ds_mega'},
    include_package_data=True,
    install_requires=[],
    zip_safe=False,
    keywords='ds_mega',
)
