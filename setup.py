from setuptools import setup

setup(
    name='linefits',
    version='0.1.0',    
    description='Package for calibration spectrum line fitting',
    url='...',
    author='Ryan Terrien',
    author_email='rterrien@carleton.edu',
    license='...',
    packages=['linefits'],
    install_requires=['numpy',
                      'scikit-image',
                      'astropy',
                      'scipy',
                      'multiprocessing_logging'],
    classifiers=[],
)