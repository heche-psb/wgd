#!/usr/bin/python3.8
# contact: heche@psb.vib-ugent.be
# https://stackoverflow.com/questions/43658870/requirements-txt-vs-setup-py

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='wgd',
    version='2.0.37',
    packages=['wgd'],
    url='http://github.com/heche-psb/wgd',
    license='GPL',
    author='Hengchi Chen',
    author_email='heche@psb.vib-ugent.be',
    description='wgd',
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=['cli'],
    include_package_data=True,
    install_requires=[
       'attrs==20.3.0',
       'biopython==1.76',
       'click==7.1.2',
       'colorama==0.4.4',
       'commonmark==0.9.1',
       'cycler==0.10.0',
       'ete3>=3.1.3',
       'fastcluster==1.1.28',
       'humanfriendly==8.2',
       'importlib-metadata==3.1.0',
       'iniconfig==1.1.1',
       'Jinja2==2.11.2',
       'joblib==0.11',
       'KDEpy==1.1.0',
       'kiwisolver==1.2.0',
       'MarkupSafe==1.1.1',
       'matplotlib==3.2.2',
       'numpy==1.19.0',
       'numexpr>=2.7.3',
       'packaging==20.4',
       'pandas<=1.4.4',
       'Pillow<=8.4.0',
       'pluggy==0.13.1',
       'plumbum==1.6.9',
       'progressbar2==3.51.4',
       'psutil==5.7.0',
       'py==1.9.0',
       'Pygments==2.7.2',
       'pyqtwebengine<5.13',
       'pyqt5<5.13',
       'pyparsing==2.4.7',
       'pytest==6.1.2',
       'python-dateutil>=2.8.2',
       'python-utils==2.4.0',
       'pytz==2020.1',
       'PyYAML==5.3.1',
       'rich==12.5.1',
       'scikit-learn==0.23.1',
       'scikit-learn-extra==0.2.0',
       'scipy<=1.5.4',
       'seaborn==0.10.1',
       'six==1.15.0',
       'threadpoolctl==2.1.0',
       'toml==0.10.2',
       'tornado==6.0.4',
       'tqdm>=4.64.1',
       'typing-extensions>=3.10.0.0',
       'zipp==3.4.0'
    ],
    entry_points='''
        [console_scripts]
        wgd=cli:cli
    ''',
)
