import numpy as np
import scipy as sp
import utilities as utils
from numpy import linalg as la

# TODO: optionally pass P(S_max) instead of deriving it internally from dN/dS 
# TODO: check that all relevant array dimensions agree 
# TODO: do the axis vectors have to be uniform linear spacing, or can they be more general (index axis must be?)
def posterior_twoband_gaussian(s_measured1, s_measured2, covmatrix, freq_bands, fluxaxis, indexaxis, s_dnds_model, dnds_model, solid_angle_arcminsq, index_prior, flat_flux_prior=False, debug=False):
  """
  Input:
  the measured fluxes in band 1 (s_measured1) and band 2 (s_measured2)
  the covariance of the Gaussian error model of the measured fluxes (covmatrix) 
  the frequencies of the two band (freq_bands)
  vectors specifying the axis of flux (fluxaxis) and spectral index axis (indexaxis) (linear in both axes)
  the differential source counts n #/Jy/deg^2 in band 1 (dnds_model) as a function of flux (s_dnds_model) 
  the solid angle in arcmin^2 over which to define the flux prior (solid_angle_arcminsq)
  the prior along the spectral index axis (index_prior)
  an optional flag that assumes a flat prior in flux instead (flat_flux_prior)
  Output:
  P(S_i1, index | S_m1, S_m2) : posterior_fluxindex
  P(S_i1, S_i2 | S_m1, S_m2) : posterior_fluxflux
  """

  freq_ratio = freq_bands[1]/freq_bands[0]
  n_fluxaxis = fluxaxis.size
  n_indexaxis = indexaxis.size
  covinv = la.inv(covmatrix)

  np.seterr(all='ignore') # TODO treat these instead
  # find the likelihood P(S_1m, S_2m | S_1i, index) assuming Gaussian noise with covmatrix
  # TODO: numpy matrix operations are ~tens of times faster than for loops, but might look awkward?
  residual_fluxindex = np.zeros((2, n_fluxaxis, n_indexaxis))
  residual_fluxindex[0,:,:] = np.repeat(fluxaxis[:, np.newaxis], n_indexaxis, 1)
  index_multiplier = (freq_ratio)**indexaxis # S_2 = S_1 (nu2/nu1)^index
  residual_fluxindex[1,:,:] = residual_fluxindex[0,:,:] * index_multiplier[np.newaxis,:]
  residual_fluxindex[0,:,:] -= s_measured1
  residual_fluxindex[1,:,:] -= s_measured2
  # ans = la.einsum('abi, ij, abj', residual_fluxindex, covinv, residual_fluxindex)? # unsupported  
  likelihood_fluxindex = np.exp(-np.sum(residual_fluxindex * np.tensordot(covinv, residual_fluxindex, (0, 0)), axis=0)/2.)

  # find the likelihood P(S_1m, S_2m | S_1i, S_2i) assuming Gaussian noise with covmatrix
  # variables can either be in terms of flux1, flux2 (named fluxflux) or flux1 and the spectral index (fluxindex)
  # TODO: can this be shortened to a matrix product eps Cov^-1 eps^T?; doesn't seem to be a way
  residual_fluxflux = np.zeros((2, n_fluxaxis, n_fluxaxis))
  residual_fluxflux[0,:,:] = np.repeat(fluxaxis[:, np.newaxis], n_fluxaxis, 1)
  residual_fluxflux[1,:,:] = residual_fluxflux[0,:,:].transpose()
  residual_fluxflux[0,:,:] -= s_measured1
  residual_fluxflux[1,:,:] -= s_measured2
  likelihood_fluxflux = np.exp(-np.sum(residual_fluxflux * np.tensordot(covinv, residual_fluxflux, (0, 0)), axis=0)/2.)

  # find P(S_1i) and make a matrix version = P(S_1i) repeating along the flux2 or spectral index direction 
  pdf_prior_flux = solid_angle_arcminsq/3600.*dnds_model*np.exp(-utils.dnds_to_ngts(s_dnds_model, dnds_model)*solid_angle_arcminsq/3600.)
  interpolant = sp.interpolate.interp1d(s_dnds_model, pdf_prior_flux, bounds_error=False, fill_value=0.)
  pdf_prior_flux_int = interpolant(fluxaxis)
  fluxprior_fluxindex = np.repeat(pdf_prior_flux_int[:, np.newaxis], n_indexaxis, 1) # is this correct, or its transpose?
  fluxprior_fluxflux = np.repeat(pdf_prior_flux_int[:, np.newaxis], n_fluxaxis, 1)
  np.seterr(all='raise')

  # find the prior
  # one can either specify the prior in flux or spectral index, and we 
  # report the posterior in both flux flux and flux index, so need to convert 
  # either of these prior conditions into the axes in the posterior
  flux_prior = np.zeros((n_fluxaxis, n_fluxaxis))
  index_prior_matrix = np.zeros((n_fluxaxis, n_indexaxis))
  ln_freq_ratio = np.log(freq_ratio)
  min_index = min(indexaxis)
  delta_index = indexaxis[1] - indexaxis[0]  
  if flat_flux_prior:
    flux_prior += 1. 
    # transform this to an index prior
    index_multiplier = (freq_ratio)**indexaxis*np.abs(ln_freq_ratio) # S_2 = S_1 (nu2/nu1)^index
    index_prior_matrix = np.repeat(fluxaxis[:, np.newaxis], n_indexaxis, 1) * index_multiplier[np.newaxis,:] # is n_indexaxis x n_fluxaxis
    # try this for comparison
    #for i in np.arange(n_fluxaxis): 
    #  index_prior_matrix[i,:] = fluxaxis[i]*(freq_ratio)**indexaxis*np.abs(ln_freq_ratio)
  else:
    for i in np.arange(n_fluxaxis):
      index_prior_matrix[i,:] = index_prior
      # find the flux prior from the index prior 
      # transform index prior into flux prior.  first, calculate effective index for each value of band2 flux.
      index_smax2 = np.log(fluxaxis/fluxaxis[i])/ln_freq_ratio
      index_indices = np.round((index_smax2 - min_index)/delta_index) # TODO: do this with http://docs.python.org/release/2.5.2/lib/bisect-example.html?
 
      whneg = (index_indices < 0)
      index_indices[np.where(whneg)] = 0
      whtb = (index_indices >= n_indexaxis)
      index_indices[np.where(whtb)] = n_indexaxis - 1

      index_to_flux = index_prior[index_indices.astype(int)]
      flux_prior[i,:] = index_to_flux/(fluxaxis*abs(ln_freq_ratio))

      flux_prior[i, np.where(whneg)] = 0.
      flux_prior[i, np.where(whtb)] = 0.

  np.seterr(all='ignore') # TODO: remove this!
  # P(S_i1, index | S_m1, S_m2) = P(S_m1, S_m2 | S_i1, index) P(S_i1, index); P(S_i1, index) = P(S_i1) P(index)
  # TODO: normalize?
  posterior_fluxindex = likelihood_fluxindex*fluxprior_fluxindex*index_prior_matrix

  # P(S_i1, S_i2 | S_m1, S_m2) = P(S_m1, S_m2 | S_i1, S_i2) P(S_i1, S_i2)
  posterior_fluxflux = likelihood_fluxflux*fluxprior_fluxflux*flux_prior
  np.seterr(all='raise') # TODO: remove this!

  if debug:
    import shelve 
    d = shelve.open("posterior.shelve")
    d["likelihood_fluxindex"] = likelihood_fluxindex
    d["fluxprior_fluxindex"] = fluxprior_fluxindex
    d["index_prior_matrix"] = index_prior_matrix
    d["likelihood_fluxflux"] = likelihood_fluxflux
    d["fluxprior_fluxflux"] = fluxprior_fluxflux
    d["flux_prior"] = flux_prior
    d["dnds_in"] = dnds_model
    d["s_in"] = s_dnds_model
    d["ngts_in"] = utils.dnds_to_ngts(s_dnds_model, dnds_model)
    d["psmax"] = pdf_prior_flux
    d["alphavec"] = indexaxis
    d["fluxvec"] = fluxaxis
    d["posterior_fluxindex"] = posterior_fluxindex
    d["posterior_fluxflux"] = posterior_fluxflux
    d.close()
    quit()

  return (posterior_fluxindex, posterior_fluxflux)
