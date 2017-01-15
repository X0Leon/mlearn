# -*- coding: utf-8 -*-

import setuptools


setuptools.setup(
    name="mlearn",
    version='0.0.3b',
    author='Leon Zhang',
    author_email='pku09zl@gmail.com',
    description='Sequence machine learning algorithms',
    keywords='python machine learning',
    url='https://github.com/X0Leon/mlearn',
    license='MIT',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
    ],
    packages=['mlearn',
              'mlearn.dhmm'],
    long_description='Implementation or wrapper machine learning algorithms for time series data.',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Software Development :: Libraries',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
    ]
)
