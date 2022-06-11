from setuptools import setup, find_packages
import re
from subprocess import run, CalledProcessError


def get_long_description(project_path):
    '''Fetch the README contents and replace relative links with absolute ones
    pointing to github for correct behaviour on PyPI.
    '''
    try:
        revision = run(
            ['git', 'describe', '--tags'],
            capture_output=True,
            check=True,
            text=True
        ).stdout[:-1]
    except CalledProcessError:
        try:
            with open('PKG-INFO', 'r') as fd:
                body = fd.read().partition('\n\n')[2]
            if body:
                return body
        except FileNotFoundError:
            revision = 'master'

    with open('README.md', 'r', encoding='utf-8') as fd:
        long_description = fd.read()

    link_root = {
        '': f'https://github.com/{project_path}/blob',
        '!': f'https://raw.githubusercontent.com/{project_path}',
    }

    def replace(mobj):
        return f'{mobj[1]}[{mobj[2]}]({link_root[mobj[1]]}/{revision}/{mobj[3]})'

    link_rexp = re.compile(r'(!?)\[([^\]]*)\]\((?!https?://|/)([^\)]+)\)')
    return link_rexp.sub(replace, long_description)


setup(
    name='zennit-crp',
    version='0.4.2',
    description='Concept Relevance Propagation and Relevance Maximization',
    author='Reduan Achtibat',
    license='BSD 3-Clause Clear License',
    long_description=get_long_description('rachtibat/zennit-crp'),
    long_description_content_type='text/markdown',
    url='https://github.com/rachtibat/zennit-crp',
    packages=find_packages(),
    install_requires=[
        'zennit',
        'torch>=1.7.0',
        'click',
    ],
    python_requires='>=3.8',

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
