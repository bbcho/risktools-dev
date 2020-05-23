import setuptools

with open("README.md","r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="risktools",
	version="0.0.1",
	author="Ben Cho",
 	license='gpl-3.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
	author_email="ben.cho@gmail.com",
	description="Python wrapper for the R package RTL",
	long_description=long_description,
	long_description_content_type="text/markdown",
	packages=setuptools.find_packages(),
	package_data={'': ['*.csv']},
	keywords = ['RTL', 'Risk', 'Tools', 'Trading', 'Crude', 'Oil'],
	url = "https://github.com/bbcho/risktools-dev",
	download_url = "https://github.com/bbcho/risktools-dev/archive/v0.0.1-beta.1.tar.gz",
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
		'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
		'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
		"Programming Language :: Python :: 3",
		"Operating System :: OS Independent",
	],
	python_requires='>=3.6',
)