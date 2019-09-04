try:
    # external dependency but (together with wheel) builds binary wheels
    from setuptools import setup, Extension
    # test if wheel is installed too:
    import wheel
except ImportError:
    # included but does not build binary wheels
    from distutils.core import setup, Extension

import sys, os
import numpy
numpy_path = numpy.__path__[0]

src_path = 'src/mcmodel/c_extension'

c_model_filename = 'mcmodel.c'

if sys.version_info.major >= 3:
    # Python 3
    with open(os.path.join(src_path, c_model_filename), 'r', encoding = 'latin1') as f:
        text = f.read()
else:
    # Python 2
    with open(os.path.join(src_path, c_model_filename), 'r') as f:
        text = f.read()

version = text.split('#define MODULE_VERSION_STRING',1)[1].split('\n',1)[0].strip()[1:-1]


CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",    
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",    
    "Topic :: Scientific/Engineering :: Physics",
]



this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()



module  = Extension('mcmodel',
                    sources=[os.path.join(src_path, fn) for fn in (c_model_filename,'dSFMT.c')],
                    define_macros=[('DSFMT_MEXP','19937')],
                    include_dirs=[src_path, numpy_path+"/core/include/numpy/"])

setup(name = 'mcmodel',
      version = version,
      author = 'Chris Petrich',
      description = 'Monte Carlo Scattering Model',
      packages = ['mcmodel_util'],
      test_suite = 'tests',
      package_dir = {'': 'src/mcmodel'},
      install_requires = ['numpy>=1.7'],
      ext_modules = [module],
      classifiers = CLASSIFIERS,
      long_description=long_description,
      long_description_content_type='text/markdown',
      )
