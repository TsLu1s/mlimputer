import setuptools
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="mlimputer",
    version="2.0.25",
    description="MLimputer - Missing Data Imputation Framework for Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TsLu1s/MLimputer",
    author="Luís Fernando da Silva Santos",
    author_email="luisf_ssantos@hotmail.com",
    license="MIT",
    classifiers=[
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

        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    py_modules=["mlimputer"],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},  
    keywords=[
        "machine learning",
        "missing data imputation",
        "data preprocessing",
        "supervised learning",
        "predictive imputation",
        "multivariate imputation",
        "random forest imputation",
        "gradient boosting imputation",
        "knn imputation",
        "automated imputation",
        "missing values",
        "data science",
        "ml pipeline",
    ],         
    install_requires=open("requirements.txt").readlines(),
)