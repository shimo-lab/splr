from setuptools import setup, find_packages

setup(name='splr',
      packages=find_packages(),
      description='Sparse plus low rank matrices library',
      url='https://github.com/shimo-lab/splr',
      install_requires=['numpy', 'scipy'])