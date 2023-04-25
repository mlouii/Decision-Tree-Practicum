from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]

with open("DecisionTreeClassifier/README.txt", "r") as f:
    long_description = f.read()
 
setup(
  name='DecisionTreeClassifier',
  version='0.0.5',
  description='A Decision Tree Classifier.',
  long_description=long_description,
  package_dir={"": "DecisionTreeClassifier"},
  packages=find_packages(where="DecisionTreeClassifier"),
  url='https://github.com/mlouii/Decision-Tree-Practicum',  
  author='Mark Lou, Jobin Joyson',
  author_email='mlou@hawk.iit.edu, jjoyson1@hawk.iit.edu',
  license='MIT', 
  classifiers=classifiers,
  keywords='decision tree', 
  install_requires=['numpy', 'pandas'] 
)