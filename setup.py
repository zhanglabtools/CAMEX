from setuptools import setup, find_packages

REQUIRED = [
    'scanpy[leiden]>=1.9.1, <=1.9.3',
    'numpy>=1.22.4, <=1.24.4',
    'pandas>=1.5.3, <=1.5.3',
    'numba>=0.56.4, <=0.56.4',
    'harmonypy>=0.0.9, <=0.0.9',
]
setup(name='CAMEX',
      author='Zhen-Hao Guo',
      author_email='guozhenhao@tongji.edu.cn',
      url='www.xxx.com',
      version='0.0.1',
      packages=find_packages(),
      install_requires=REQUIRED,
      )
