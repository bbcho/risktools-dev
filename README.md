# pyRTL

Python wrapper for R library RTL found at https://github.com/risktoollib/RTL

NOTES: 
* Only works with rpy2 version >= 3.0. Verified working with version 3.2.6
* All datetimes are given in UTC
* Verified working for RTL version 0.1.1

Must have the following R libraries already installed on host machine:

* RTL
* devtools
* tidyverse
* tidyquant
* Quandl

And the following Python libaries:

* rpy2
* pandas
* numpy
* tzlocal
