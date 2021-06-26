Should be a simple ML excerise to map successful clubs against LSOA (Lower-layer Super Output Areas), using the various components of the Indicies of Multiple Deprivation

Current version reduces 32k LSOA to around 23k LSOA, so it appears to be doing something.

Stripping out IMD scores (as that's calculated from the other scores), and population data, we get down to 18.5k LOSA's.

Next challenge is to see if weighting can further reduce possible sites...

# Current issue
Seems to be working

# dependancies
scikit-learn
pandas