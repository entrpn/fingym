from distutils.core import setup
setup(
  name = 'fingym',
  packages = ['fingym'],
  version = '0.2',
  license='apache-2.0',    
  description = 'A tool for developing reinforcement learning algorithms focused in stock prediction',
  author = 'Juan Acevedo',
  author_email = 'entrpn@gmail.com',
  url = 'https://github.com/entrpn/fingym/', 
  download_url = 'https://github.com/entrpn/fingym/archive/v0.1.tar.gz',
  keywords = ['stock-market', 'stock-price-prediction', 'python','reinforcement-learning-environments','reinforcement-learning','reinforcement-learning-agents','artificial-intelligence','q-learning','evolution-strategies'],   # Keywords that define your package best
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