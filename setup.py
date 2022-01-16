import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="risktools",
    version="0.2.1",
    author="Ben Cho",
    license="gpl-3.0",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    author_email="ben.cho@gmail.com",
    description="Python implementation of the R package RTL v0.1.8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.csv", "*.json", "*.geojson"]},
    keywords=[
        "RTL",
        "Risk",
        "Tools",
        "Trading",
        "Crude",
        "Oil",
        "Refinery",
        "Refined Products",
        "Products",
    ],
    url="https://github.com/bbcho/risktools-dev",
    # download_url="https://github.com/bbcho/risktools-dev/archive/v0.5.0-beta.1.tar.gz",
    project_urls={
        # "Bug Tracker": "https://github.com/statsmodels/statsmodels/issues",
        "Documentation": "https://risktools.readthedocs.io/en/latest/",
        # "Source Code": "https://github.com/statsmodels/statsmodels",
    },
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "quandl",
        "scikit-learn",
        "arch",
        "scipy",
        "statsmodels",
        "seaborn",
        "pandas_datareader",
    ],
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
