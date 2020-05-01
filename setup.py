from setuptools import setup, find_packages, find_namespace_packages

import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fingym'))

setup(
  name = 'fingym',
  version = '0.5',
  license='apache-2.0',    
  description = 'A tool for developing reinforcement learning algorithms focused in stock prediction',
  author = 'Juan Acevedo',
  author_email = 'entrpn@gmail.com',
  url = 'https://github.com/entrpn/fingym/', 
  download_url = 'https://github.com/entrpn/fingym/archive/v0.5.tar.gz',
  keywords = ['stock-market', 'stock-price-prediction', 'python','reinforcement-learning-environments','reinforcement-learning','reinforcement-learning-agents','artificial-intelligence','q-learning','evolution-strategies'],
  packages=find_namespace_packages(),
  package_data={'fingym': ['data/*.csv']},
  install_requires=[
          'pandas',
          'numpy',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)