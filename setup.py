from setuptools import setup, find_packages

setup(
    name='neurokin',
    version='0.0.1',
    description='package to support neural and data analysis',
    url='https://github.com/WengerLab/neurokin',
    author='Elisa L. Garulli',
    author_email='e.garulli@charite.de',
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=['c3d==0.5.2',
                      'fooof==1.0.0',
                      'tdt==0.5.3',
                      'numpy==1.24.4',
                      'pandas==2.1.4',
                      'scipy==1.11.4',
                      'matplotlib==3.7.3',
                      'typeguard==4.4.2',
                      'pyyaml==6.0.2'
                      ],
extras_require={
        "dev": ["pytest"],
        "docs": ["sphinx", "nbsphinx", "sphinxawesome-theme", "pygments"]
    },

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)