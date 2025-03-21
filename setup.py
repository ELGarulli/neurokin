from setuptools import setup, find_packages

setup(
    name='neurokin',
    version='0.0.1',
    description='package to support neural and data analysis',
    url='https://github.com/WengerLab/neurokin',
    author='Elisa L. Garulli',
    author_email='e.garulli@charite.de',
    license='BSD 2-clause',
    packages=find_packages('neurokin'),
    package_dir={'': 'neurokin'},
    install_requires=['c3d==0.5.1',
                      'fooof==1.0.0',
                      'tdt==0.5.3',
                      'numpy~=1.26.4',
                      'pandas>=2.0',
                      'scipy~=1.8.0',
                      'matplotlib~=3.5.1',
                      'scikit-learn~=1.1.3',
                      'typeguard'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)