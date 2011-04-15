import process_catalog as process
import self_describing as sd

# SPT deboosting parameters
# num_X is number of grid points
# cov_calbeam is the covariance contribution due to beam and calibration uncertainty 
# plus covariance from background sources (found from simulation)
param = {
  "percentiles":[0.16, 0.5, 0.84],
  "num_flux":800,
  "num_alpha":800,
  "prior_alpha":[-5.,5.],
  "range_alpha":[-5.,5.],
  "cov_calbeam":[[0.00165700, 0.00129600], [0.00129600, 0.00799300]],
  "freq1":152.,
  "freq2":219.5,
  "omega_prior":1.,
  "flux1name":"flux150",
  "flux2name":"flux220",
  "sigma1name":"sigma150",
  "sigma2name":"sigma220",
  "keyfield_name":"index",
  "catalog_filename":"../data/source_catalog_vieira09_3sigma.dat",
  "neglect_calerror" : False,
  "verbose" : True
}

# rename some fields from the self-describing input ascii catalog
translate={'ID':param['keyfield_name'], 'S_150':param['flux1name'], 'noise_150':param['sigma1name'], 'S_220':param['flux2name'], 'noise_220':param['sigma2name']}
input_catalog = sd.load_selfdescribing_numpy(param['catalog_filename'], swaps=translate, verbose=param['verbose'])

# convert flux150, sigma150, flux220, sigma220 from mJy to Jy
input_catalog[:][param['flux1name']] /= 1000.
input_catalog[:][param['sigma1name']] /= 1000.
input_catalog[:][param['flux2name']] /= 1000.
input_catalog[:][param['sigma2name']] /= 1000.

augmented_catalog = process.process_ptsrc_catalog_alpha(input_catalog, param)
