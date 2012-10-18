#!/usr/bin/env python
r"""Main call to the two-band posterior flux code"""
from optparse import OptionParser
import self_describing as sd
import compare_catalogs as cc
import utilities as utils
import process_catalog
import shelve
import os

def wrap_process_catalog(params, translate, print_only=False):
    r"""Based on the ini file, load the catalog and general parameters
    """
    (root, extension) = os.path.splitext(run_param['catalog_filename'])

    if extension == ".dat":
        print "opening a catalog in the self-describing text format"
        catalog = sd.load_selfdescribing_numpy(run_param['catalog_filename'],
                                               swaps=trans_table,
                                               verbose=run_param['verbose'])

    if extension == ".pickle":
        import pickle
        import catalog
        print "opening a catalog as a pickled catalog object"
        catalog = pickle.load(open(run_param['catalog_filename'], "r"))

    if not print_only:
        augmented_catalog = process_catalog.process_ptsrc_catalog_alpha(
                                                        catalog, params)

        outputshelve = shelve.open(params['augmented_catalog'], flag="n",
                                   protocol=-1)
        outputshelve.update(augmented_catalog)
        outputshelve.close()
    else:
        print catalog


if __name__ == '__main__':
    r"""main command-line interface
    python bayes_flux.py data/spt_catalog.ini | tee catalog_run.log
    python bayes_flux.py data/spt_catalog.ini -c | tee comp.log
    enscript -2Gr comp.log -p comprun.ps
    """

    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")

    parser.add_option("-c", "--compare",
                      action="store_true",
                      dest="compareflag",
                      default=False,
                      help="Compare with a published catalog")

    parser.add_option("-p", "--print",
                      action="store_true",
                      dest="print_only",
                      default=False,
                      help="Just print the catalog and exit")

    (options, args) = parser.parse_args()
    optdict = vars(options)

    if len(args) != 1:
        parser.error("wrong number of arguments")

    print options
    print args

    params = utils.iniparse(args[0], flat=True)

    translate = {'ID': params['keyfield_name'],
                 'S_150': params['flux1name'],
                 'noise_150': params['sigma1name'],
                 'S_220': params['flux2name'],
                 'noise_220': params['sigma2name']}

    if not optdict['compareflag']:
        print "augmenting the catalog with posterior distrubtions"
        wrap_process_catalog(params, translate,
                             print_only=optdict['print_only'])

    else:
        print "comparing catalogs"
        cc.compare_catalogs(params, translate)
