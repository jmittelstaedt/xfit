try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.md') as f:
    readme = f.read()

setup(
    name='xfit',
    version='0.1',
    packages=['xfit'],
    description='Curve fitting for xarray.',
    long_description=readme,
    install_requires = [
        "xarray",
        "numpy",
        "scipy",
        "matplotlib"
    ]
)
