import sys
import os
from setuptools import setup


if sys.version_info < (3, 9):
    sys.stderr.write(
        f'You are using Python '
        + "{'.'.join(str(v) for v in sys.version_info[:3])}.\n\n"
        + 'aydin only supports Python 3.9 and above.\n\n'
        + 'Please install Python 3.9 using:\n'
        + '  $ pip install python==3.9\n\n'
    )
    sys.exit(1)

with open(os.path.join('requirements', 'default.txt')) as f:
    default_requirements = [
        line.strip() for line in f if line and not line.startswith('#')
    ]

INSTALL_REQUIRES = default_requirements
REQUIRES = []

# Handle pyopencl
# os.system("pip install -r " + os.path.join('requirements', 'pyopencl.txt'))

setup(
    name='aydin',
    # packages=['aydin'],
    # use_scm_version=True,
    version="0.0.6",
    # setup_requires=['setuptools_scm'],
    url='https://github.com/royerlab/aydin',
    install_requires=INSTALL_REQUIRES,
    py_modules=['aydin'],
    entry_points={
        'console_scripts': ['aydin=aydin.cli.cli:cli']
        },
    license='BSD 3-Clause',
    license_file='LICENSE.txt',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
      ],
)
