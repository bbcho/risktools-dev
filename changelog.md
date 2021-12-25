# Dec 24, 2021
- ir_ud_df changed to add index with quandl codes. First line is today and is indexed at "0". Change made to align with RTL. ALso added capability to fix start and end dates to make testing easier in the future
- trade_stats omega ratio no longer annualized to align with RTL
- added regex=True to all replace methods for pandas
- updated all datasets
- changed test for rolladjust function to test for a single contract instead of all
- changed swap_irs mat_date implementation so that if it doesn't fall evenly on freq, take last date before maturity
- removed chart_spreads.json from test_chart_spreads as df is empty and caused error
- removed requirement for geojson - moved geojson files to a function that downloads the files as required based on a url. geojson only required for this function so added imports as a try/except chunk. Geojson required C++ library gdal which can cause issues if you don't have sudo access.
- only works for >-3.7 because of arch packge