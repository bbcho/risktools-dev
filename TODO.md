# TODO

## Testing
* figure out a way so that the prompt_beta results match RTL::promptBeta exactly.
    Issue is that both use a linear regression to determine the beta, but they're
    returning slightly different betas (on the order of 0.001). Need to determine if this is because of an error in my code or because of differences in implementation in linear regression models between R and Python.

## Enhancements
* npv: remove need for live ir curve if using fixed yield. Just need to get maturities

## 0.1.8 update
- since need to add discountcurve function from RQuantlib - see Rcpp code in package. not in Quantlib base as it combines a bunch of different methods together - convert Rcpp code to Python using quantlib for python