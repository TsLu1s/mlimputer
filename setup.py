import setuptools
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="mlimputer",
    version="1.0.80",
    description="MLimputer - Missing Data Imputation Framework for Machine Learning",
    long_description=long_description,      
    long_description_content_type="text/markdown",
    url="https://github.com/TsLu1s/MLimputer",
    author="Luís Santos",
    author_email="luisf_ssantos@hotmail.com",
    license="MIT",
    classifiers=[
        # Indicate who your project is intended for
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Customer Service",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Telecommunications Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",

        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",

        "Programming Language :: Python :: 3.8",
    ],
    py_modules=["mlimputer"],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},  
    keywords=[
        "data science",
        "machine learning",
        "data preprecessing",
        "null imputation",
        "predictive null imputation",
        "multiple null imputation",
        "automated machine learning",
    ],           
    install_requires=open("requirements.txt").readlines(),
)
