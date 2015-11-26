from setuptools import setup, find_packages 
from codecs import open  
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='stacked_generalization',
    packages=find_packages(),
    version='0.0.0',
    description='Python implementation of stacked generalization. Plays nicely with sklearn',
    long_description=long_description,    
    url='https://github.com/dustinstansbury/stacked_generalization',
    author='Dustin Stansbury',
    author_email='https://dustin.stansbury@gmail.com',
    keywords='machine learning, stacked generalization, ensemble methods, classification algorithms',

    install_requires=['numpy>1.6.1',
                      'scipy',
                      'scikit-learn'
                      ],
)
