`bayes_flux`
============
`bayes_flux` is a Bayesian flux reconstruction for astronomical surveys in two bands.

Assumptions:
------------
* The abundance (dN/dS) for flux S is known ahead of time (for the prior).
* Noise terms from the instrument (calibration and thermal) and background sources at confusion are described by only a covariance.

Usage:
------
1. adapt `load_catalog.py` to load your source catalogs
2. calculate and plug in your survey-specific covariance between bands
3. adapt `source_count_models.py` and `process_catalogs.py` to reflect the source count priors

Disclaimer: this is active development version. We provide default examples to test the code against values in literature, but you should look critically at what the code does before using it in published work. Find official comparison catalogs on: `http://pole.uchicago.edu/public/data/vieira09/index.html`

Primary citation:
------------------
For more information on the algorithm, and when using this code for published work, cite:
"A method for individual source brightness estimation in single- and multi-band millimeter-wave data"
T. M Crawford, E. R Switzer, W. L Holzapfel, C. L Reichardt, D. P Marrone, J. D Vieira
The Astrophysical Journal, 718:513–521, 2010 July 20.

Abstract:
We present a method of reliably extracting the flux of individual sources from millimeter/submillimeter (mm/submm) sky maps in the presence of noise and a steep source population. The method is an extension of a standard Bayesian procedure in the mm/submm literature, developed to account for the known bias incurred when attempting to measure source fluxes for a population in which there are many more faint sources than bright ones. As in the standard method, the prior applied to source flux measurements is derived from an estimate of the source counts as a function of flux, dN/dS. The key feature of the new method is that it enables reliable extraction of properties of individual sources, which previous methods in the literature do not. We first present the method for extracting individual source fluxes from data in a single observing band, then we extend the method to multiple bands, including prior information about the spectral behavior of the source population(s). The multi-band estimation technique is particularly relevant for classifying individual sources into populations according to their spectral behavior. We find that proper treatment of the correlated prior information between observing bands is key to avoiding significant biases in estimations of multi-band fluxes and spectral behavior, biases which lead to significant numbers of misclassified sources. We test the single- and multi-band versions of the method using simulated observations with observing parameters similar to that of the South Pole Telescope data used in Vieira et al., 2009.

Counts models:
--------------
In addition to the main code, cite counts models used in the prior:

Title: Radio and millimeter continuum surveys and their astrophysical implications
Authors: de Zotti, Gianfranco; Massardi, Marcella; Negrello, Mattia; Wall, Jasper
Publication: The Astronomy and Astrophysics Review, Volume 18, Issue 1-2, pp. 1-65, 2/2010
ADS Bibliographic Code: `2010A&ARv..18....1D`
Website: `http://web.oapd.inaf.it/rstools/srccnt/srccnt_tables.html`

Title: Modeling the evolution of infrared galaxies: a parametric backward evolution model
Authors: Béthermin, M.; Dole, H.; Lagache, G.; Le Borgne, D.; Penin, A.
Publication: Astronomy & Astrophysics, Volume 529, id.A4, 5/2011
ADS Bibliographic Code: `2011A&A...529A...4B`
Website: `http://www.ias.u-psud.fr/irgalaxies/model.php`

Note that the counts models here do not coincide with those used in the prior of SPT source catalog paper, Vieira 2009 (which were private communications and need to be cleared for publication with the code), so the deboosted fluxes will be a little different, especially at low flux.

Suggestions for calculating `cov_calibration` for other surveys:
--------------------------------------------------------------
* Use flux recovery simulations to estimate the calibration/beam error covariance. (a set of PSFs drawn from the error model implies scatter in flux; can do in Fourier/multipole space)
* Find the covariance of background fluctuations by applying the matched finding/photometry filter, and calculating the covariance between bands. The map could either be: 1) an empirical noise estimate (e.g. difference map) + CMB etc. realization + source realization up to the threshold, 2) the real map with sources removed down to the threshold.

Major TODOs of the current code and wish list:
----------------------------------------------
* Extend to  greater than 2 bands.
* Write some generic code to read the IR and radio source count models: given name and freq find closest band and transform the counts if necessary
* Add `http://cmbr.phas.ubc.ca/model/`, and other counts models cited on `http://www.ias.u-psud.fr/irgalaxies/model.php`
* Code to plot/compare counts models
* Code to simulate surveys under basic noise and source population assumptions
* Write unit tests for the major functions
* Pass in a flux prior from simulations instead of the analytic form
* Permit more general dN/dS without linear spacing

Authors (software):
-------------------
The original internal collaboration code was written by T. Crawford and E. Switzer, and the refactoring into python here is by E. Switzer.
