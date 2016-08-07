from setuptools import setup

setup(
    name='panel_econometrics',
    version='0.1',
    description='Econometrics for Panel Data',
    long_description=open('README.md').read(),
    author='Nicolas HENNETIER',
    author_email='nicolashennetier2@gmail.com',
    packages=['panel_econometrics'],
    requires=['numpy', 'pandas', 'scipy', 'matplotlib']
)