
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Quick Parallel',
    'author': 'Lucas David',
    'url': '',
    'author_email': 'lucasolivdavid@gmail.com',
    'version': '0.1',
    'install_requires': ['numpy', 'nose'],
    'packages': ['quick_parallel'],
    'scripts': [],
    'name': 'quick_parallel'
}

setup(**config)
