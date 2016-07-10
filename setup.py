import io
import sys

from setuptools import find_packages, setup

with io.open('conx/__init__.py', encoding='utf-8') as fid:
    for line in fid:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break

setup(name='conx',
      version=version,
      description='Neural network library on Theano',
      long_description=open('README.md', 'rb').read().decode('utf-8'),
      author='Douglas S. Blank',
      author_email='dblank@cs.brynmawr.edu',
      url='https://github.com/Calysto/conx',
      install_requires=['IPython>=3.0'],
      packages=find_packages(include=['conx', 'conx.*']),
      classifiers=[
          'Framework :: IPython',
          'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
          'Programming Language :: Python :: 3',
      ]
)
