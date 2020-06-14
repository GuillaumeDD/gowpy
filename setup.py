from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="gowpy",
    version="0.2.0",
    description="A very simple graph-of-words library for NLP",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Guillaume Dubuisson Duplessis",
    author_email="guillaume@dubuissonduplessis.fr",
    url="https://github.com/GuillaumeDD/gowpy.git",
    packages=find_packages(exclude="tests"),
    license="new BSD",
    install_requires=required,
    include_package_data=True,
    python_requires=">=3.6",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)
