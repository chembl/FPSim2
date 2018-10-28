from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np    

setup(
    name='FPSim2',
    version='0.12',
    author='Eloy Félix',
    author_email='eloyfelix@gmail.com',
    description='Simple package for fast molecular similarity searching',
    url='https://github.com/eloyfelix/FPSim2',
    license='MIT',
    packages=[
        'FPSim2',
        'FPSim2.io'
        ],
    long_description=open('README.md').read(),
    install_requires=[
        'tables>=3.4.4',
        'numpy>=1.15.2'
   ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    ext_modules=[
        Extension('FPSim2.FPSim2lib',
                    sources=['FPSim2/FPSim2lib.pyx'],
                    extra_compile_args=['-march=native', '-funroll-loops'],
                    language='c',
                    include_dirs=[np.get_include()]),
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Chemistry'
        ],
    cmdclass = {'build_ext': build_ext}
)
