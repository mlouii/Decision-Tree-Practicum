from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='DecisionTreeClassifier',
  version='0.0.1',
  description='A Decision Tree Classifier.',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='Mark Lou, Jobin Joyson',
  author_email='mlou@hawk.iit.edu, jjoyson1@hawk.iit.edu',
  license='MIT', 
  classifiers=classifiers,
  keywords='decision tree', 
  packages=find_packages(),
  install_requires=['numpy', 'pandas', 'random', 'time', 'json', 'urllib.parse', 'webbrowser', 'abc', 'collections'] 
)