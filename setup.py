from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np
import platform


c_comp_args = ['-O3', '-msse4.2']

if platform.system() == 'Darwin':
    c_comp_args += ['-mmacosx-version-min=10.9']
elif platform.system() == 'Linux':
    pass
elif platform.system() == 'Windows':
    raise Exception('FPSim2 is not working in Windows platforms yet.')
else:
    # so it can be installed in any platform like raspberry pi
    # via python setup.py install 
    c_comp_args = ['-march=native']

setup(
    name='FPSim2',
    version='0.0.3',
    author='Eloy Félix',
    author_email='eloyfelix@gmail.com',
    description='Simple package for fast molecular similarity searching',
    url='https://github.com/eloyfelix/FPSim2',
    license='MIT',
    packages=[
        'FPSim2',
        'FPSim2.io'
        ],
    long_description=open('README.md', encoding='utf-8').read(),
    install_requires=[
        'tables>=3.4.4',
        'numpy>=1.14'
        ],
    ext_modules=[
        Extension('FPSim2.FPSim2lib',
                    sources=['FPSim2/FPSim2lib.pyx'],
                    extra_compile_args=c_comp_args,
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
