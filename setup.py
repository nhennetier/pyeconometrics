from setuptools import setup

setup(
    name='mord',
    version='0.1',
    description='Econometrics for Panel Data',
    long_description=open('README.md').read(),
    author='Nicolas HENNETIER',
    author_email='nicolashennetier2@gmail.com',
    packages=['mord'],
    requires=['numpy', 'pandas', 'scipy', 'matplotlib']
)