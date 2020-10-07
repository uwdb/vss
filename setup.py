from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='vfs',
    version='0.1.0',
    description='A video file system',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/anonymousrepository272/anonymousrepository',
    author='Anonymous',
    author_email='anonymous@anonymous.com',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='video, encoding, file system',
    #package_dir={'': 'vfs'},
    install_requires=required,
    packages=find_packages(exclude=('src', 'inputs', 'data')),

    python_requires='>=3, <4',

    entry_points={
        'console_scripts': [
            'vfs=vfs.entrypoint:main',
        ],
    }
)