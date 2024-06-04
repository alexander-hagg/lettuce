#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import versioneer

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', "torch>=1.2", "numpy", "matplotlib", "pyevtk"]

setup_requirements = ['pytest-runner', 'pytest']

setup(
    author="Andreas Kraemer",
    author_email='kraemer.research@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Lattice Boltzmann Python GPU",
    entry_points={
        'console_scripts': [
            'lettuce=lettuce.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    package_data={'lettuce.native': ['lettuce/native/template/setup.py']},
    include_package_data=True,
    keywords='lettuce',
    name='lettuce',
    packages=find_packages(include=[
        'lettuce',
        'lettuce.ext',
        'lettuce.ext._boundary',
        'lettuce.ext._collision',
        'lettuce.ext._equilibrium',
        'lettuce.ext._flows',
        'lettuce.ext._force',
        'lettuce.ext._reporter',
        'lettuce.ext._stencil',
        'lettuce.native',
        'lettuce.native.ext',
        'lettuce.native.ext._boundary',
        'lettuce.native.ext._collision',
        'lettuce.native.ext._equilibrium',
        'lettuce.native.ext._force',
        'lettuce.util']),
    setup_requires=setup_requirements,
    test_suite='tests',
    url='https://github.com/lettucecfd/lettuce',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
