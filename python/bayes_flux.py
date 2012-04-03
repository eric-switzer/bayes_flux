#!/usr/bin/env python
r"""Main call to the two-band posterior flux code"""
from optparse import OptionParser
import self_describing as sd
import utilities as utils
import process_catalog
import shelve


def wrap_process_catalog(inifile):
    r"""Based on the ini file, load the catalog and general parameters
    """
    params = utils.iniparse(inifile, flat=True)

    translate = {'ID': params['keyfield_name'],
                 'S_150': params['flux1name'],
                 'noise_150': params['sigma1name'],
                 'S_220': params['flux2name'],
                 'noise_220': params['sigma2name']}

    catalog = sd.load_selfdescribing_numpy(params['catalog_filename'],
                                           swaps=translate,
                                           verbose=params['verbose'])

    augmented_catalog = process_catalog.process_ptsrc_catalog_alpha(
                                                    catalog, params, translate)

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

    if len(args) != 1:
        parser.error("wrong number of arguments")

    print options
    print args

    wrap_process_catalog(args[0])


if __name__ == '__main__':
    main()
