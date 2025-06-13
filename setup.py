from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='spline_toolkit',
    version='25.1.0',  # Year-based versioning: Year.Major.Minor
    description='A toolkit for spline-based geometric modeling, including Hermite spline and compound Hermite curves',
    author='Sean Auer',
    author_email='sean@seanauer.com',
    url='https://github.com/SeanAuer/spline_toolkit',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)