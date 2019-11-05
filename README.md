# pyRTL

NOTE: Only works with rpy2 version < 3.0. Verified working with version 2.9.5.

Need to upgrade to 3.X

Python wrapper for R library RTL found at https://github.com/risktoollib/RTL

Must have the following R libraries already installed on host machine

* RTL
* devtools
* tidyverse
* tidyquant
* Quandl

And the following Python libaries

* rpy2
* pandas
* numpy

So far the following functions from RTL have been added

* ir_df_us
* simGBM
* simOU
* simOUJ
* fitOU
* npv
