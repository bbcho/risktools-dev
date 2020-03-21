import setuptools

with open("README.md","r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="pyRTL",
	version="0.0.1",
	author="Ben Cho",
	author_email="ben.cho@gmail.com",
	description="Python wrapper for the R package RTL",
	long_description=long_description,
	long_description_content_type="text/markdown",
	packages=setuptools.find_packages(),
	package_data={'': ['*.csv']},
	install_requires=[
          'pandas',
          'numpy',
          'matplotlib',
          'plotly',
          'rpy2>=3.2.6',
          'tzlocal'
    ],
	include_package_data=True,
	classifiers=[
		"Programming Language :: Python :: 3",
		"Operating System :: OS Independent",
	],
	python_requires='>=3.6',
)