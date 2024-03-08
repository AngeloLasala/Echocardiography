from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='echocardiography',
    version='1.0',
    description='Regression and Diffusion model for ecocardio data',
    author='Angelo Lasala',
    author_email='Lasala.Angelo@santannapisa.it',
    packages=find_packages(),
    install_requires=requirements,
)