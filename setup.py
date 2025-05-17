from setuptools import setup, find_packages
from argparse import ArgumentParser

def read_requirements(path: str) -> str:
  """
  Reads requirements.txt document and returns the contents.

  @params path (srt): Path to requirements.txt file.
  
  @returns (str): String containing the required packages.
  """
  with open(path, 'r', encoding='utf-8-sig') as file:
    data = file.read()
  return data

def read_readme(path: str) -> str:
  """
  Reads Readme.md document and returns the content.

  @params path (srt): Path to Readme.md file.
  
  @returns (str): String containing the Readme description.
  """
  with open(path, 'r') as file:
    data = file.read()
  return data

setup(
  name = 'autograd_engine',
  version = '0.0.14',
  author = 'Sumit Kumar',
  author_email = 'sumitdvlp@gmail.com',
  maintainer = 'Sumit Kumar',
  url = 'https://github.com/sumitdvlp/autograd_fromscratch_pytorchbase',
  python_requires = '>=3.0',
  install_requires = read_requirements('requirements.txt').split('\n'),
  description = 'Autograd engine build from scratch using pytorch tensor as data holder.',
  license = 'MIT',
  keywords = 'autograd deep-learning machine-learning ai pytorch python',
  packages = find_packages(),
  long_description = read_readme('README.md'),
  long_description_content_type='text/markdown'
)