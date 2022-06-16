from setuptools import setup, find_packages

setup(
    name='openquantumcomputing',
    version='0.2',
    license='GNU General Public License v3.0',
    author="Franz Georg Fuchs",
    author_email='franzgeorgfuchs@gmail.com',
    description='Tools for quantum computing',
    long_description=open('README.md').read(),
    url='https://github.com/OpenQuantumComputing/OpenQuantumComputing',
    packages=['openquantumcomputing'],
    keywords='quantum computing, qaoa, qiskit',
    install_requires=['qiskit','numpy','scipy'],
)
