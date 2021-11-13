import sys
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

# with open(os.path.join('requirements', 'default.txt')) as f:
#     default_requirements = [
#         line.strip() for line in f if line and not line.startswith('#')
#     ]
#
# INSTALL_REQUIRES = default_requirements
# REQUIRES = []

# Handle pyopencl
# os.system("pip install -r " + os.path.join('requirements', 'pyopencl.txt'))

setup(
    use_scm_version={"write_to": "aydin/_version.py"},
    setup_requires=['setuptools_scm'],
    entry_points={'console_scripts': ['aydin=aydin.cli.cli:cli']},
)
