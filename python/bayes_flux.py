#!/usr/bin/env python
r"""Main call to the two-band posterior flux code"""
from optparse import OptionParser
import self_describing as sd
import compare_catalogs as cc
import utilities as utils
import process_catalog
import shelve


def wrap_process_catalog(params, translate):
    r"""Based on the ini file, load the catalog and general parameters
    """
    catalog = sd.load_selfdescribing_numpy(params['catalog_filename'],
                                           swaps=translate,
                                           verbose=params['verbose'])

    augmented_catalog = process_catalog.process_ptsrc_catalog_alpha(
                                                    catalog, params)

    outputshelve = shelve.open(params['augmented_catalog'], flag="n")
    outputshelve.update(augmented_catalog)
    outputshelve.close()


def main():
    r"""main command-line interface"""

    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")

    parser.add_option("-c", "--compare",
                      action="store_true",
                      dest="compareflag",
                      default=False,
                      help="Compare with a published catalog")

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
        wrap_process_catalog(params, translate)
    else:
        print "comparing catalogs"
        cc.compare_catalogs(params, translate)


if __name__ == '__main__':
    main()
