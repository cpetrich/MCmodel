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

module  = Extension('mcmodel',
                    sources=[os.path.join(src_path, fn) for fn in (c_model_filename,'dSFMT.c')],
                    define_macros=[('DSFMT_MEXP','19937')],
                    include_dirs=[src_path, numpy_path+"/core/include/numpy/"])

setup(name = 'MCmodel',
      version = version,
      author = 'Chris Petrich',
      description = 'Monte Carlo Scattering Model',
      packages = ['mcmodel_util'],
      test_suite = 'tests',
      package_dir = {'': 'src/mcmodel'},
      install_requires = ['numpy>=1.7'],
      ext_modules = [module] )
