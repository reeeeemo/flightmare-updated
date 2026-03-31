import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

setup(
    name='rpg_baselines',
    version='0.0.2',
    author='Robert Oxley',
    author_email='robert@oxley.ca',
    description='Flightmare-Updated: An Quadrotor Simulator.',
    long_description='',
    install_requires=['gymnasium', 'ruamel.yaml',
                      'numpy', 'stable_baselines3',
                      'tensorboard', 'ultralytics'],
    packages=find_packages(),
)
