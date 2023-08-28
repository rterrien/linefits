#import hpfspec2
# this version is modified with ability to treat the reference LFC and Etalon files in the same way
# (in NEID, the LFC mode index file is used to id the modes, but this is not needed)
from linefits import fitLib
import sys
import os
import numpy as np 
import argparse
try:  # works in py2/3
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
from collections import OrderedDict
import logging
import scipy.optimize
import scipy.constants
import astropy.stats
import copy
from astropy.io import fits
import datetime
import glob
from multiprocessing import Pool
from multiprocessing_logging import install_mp_handler
import astropy.stats


def parse_args(raw_args=None):
    """ Command line argument parser
    
    This is a customized argparser for the naive wavecal module
    """
    parser = argparse.ArgumentParser(description="Line fitting")
    #parser.add_argument('InputDate', type=str,
    #                    help="Input date e.g. 20201230")
    parser.add_argument('ConfigFile',type=str,help='Path to config file for this run')
    parser.add_argument('TargetFile',type=str,help='Path to spectrum file, or file containing list of such files')
    parser.add_argument('--logfile', type=str, default=None,
                        help="Log Filename to write logs during the run")
    parser.add_argument("--loglevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO', help="Set the logging level")
    parser.add_argument('--noCPUs', type=int, default=1,
                        help='number of CPUS to be used for this run')
    #parser.add_argument("--filelist", type=str,default=None,
    #                    help="Point to list of files to do line fits on, supercede L1 folder in config file")
    parser.add_argument('--filelist', action='store_true')
    parser.add_argument('--no-filelist', dest='filelist', action='store_false')
    parser.set_defaults(feature=False)

    args = parser.parse_args(raw_args)
    return(args)

def parse_str_to_types(string):
    """ Converts string to different object types they represent.
    Supported formats: True,Flase,None,int,float,list,tuple"""
    if string == 'True':
        return True
    elif string == 'False':
        return False
    elif string == 'None':
        return None
    elif string.lstrip('-+ ').isdigit():
        return int(string)
    elif string == '':
        return ''
    elif (string[0] in '[(') and (string[-1] in ')]'): # Recursively parse a list/tuple into a list
        if len(string.strip('()[]')) == 0:
            return []
        else:
            return [parse_str_to_types(s) for s in string.strip('()[]').split(',')]
    else:
        try:
            return float(string)
        except ValueError:
            return string
        

def create_configdict_from_file(configFilename,listOfConfigSections=None,flattenSections=True):
    """ Returns a configuration object as a dictionary by loading the config file.
        Values in the config files are parsed appropriately to python objects.
    Parameters
    ----------
    configFilename : str
                    File name of the config file to load
    listOfConfigSections : list (default:None)
                    Only the sections in the listOfConfigSections will be loaded to the dictionary.
                    if listOfConfigSections is None (default), all the sections in config file will be loaded.
    flattenSections: (bool, default=True)
                    True: Flattens the sections in the config file into a single level Config dictionary.
                    False: will return a dictionary of dictionaries for each section.
    Returns
    -------
    configDictionary : dictionary
                    if `flattenSections` is True (default): Flattened {key:value,..} dictionary is returned
                    else: A dictionary of dictionary is returned {Section:{key:Value},..}
    """
    configLoader = ConfigParser.ConfigParser()
    configLoader.optionxform = str  # preserve the Case sensitivity of keys
    with open(configFilename) as cfgFile:
        configLoader.read_file(cfgFile)

    # Create a Config Dictionary
    config = {}
    if isinstance(listOfConfigSections,str): # Convert the single string to a list
        listOfConfigSections = [listOfConfigSections]
    elif listOfConfigSections is None:
        listOfConfigSections = configLoader.sections()

    for configSection in listOfConfigSections:
        if flattenSections:  # Flatten the sections
            for key,value in configLoader.items(configSection):
                config[key] = parse_str_to_types(value)
        else:
            subConfig = {}
            for key,value in configLoader.items(configSection):
                subConfig[key] = parse_str_to_types(value)
            config[configSection] = subConfig
    return config


def pool_measure_centroids_order(cmdset):
    if 'return_full' in cmdset.keys():
        return_full = cmdset['return_full']
    else:
        return_full = False

    if 'variance' in cmdset.keys():
        variance = cmdset['variance']
    else:
        variance = None

    out = measure_centroids_order(cmdset['xx'],cmdset['flux'],cmdset['window_centers'],cmdset['Config'],
                                  variance=variance,return_full=return_full)
        #out = measure_centroids_order(cmdset['xx'],cmdset['flux'],cmdset['window_centers'],
        #                              sigma=None,fitfunction=cmdset['fit_function'],
        #                              fit_width_pix=cmdset['fit_width'],
        #                              basic_window_check=cmdset['basic_window_check'])
    return out

def measure_centroids_order(xx_pix, flux, window_centers, Config, variance=None, return_full=False):
    """ Measure the centroids of all lines in an order.
    
    Using a naive approach (same simple function for each line, weighted least squares,
    a single round of fitting), fit the line centroids along a spectral order.
    This is a simple wrappper to the wavecalLib.fit_lines_order() method
    
    Parameters
    ----------
    flux : ndarray
        flux for each pixel
    window_centers : ndarray
        centers for each fitting window (estimated beforehand)
    Config : dict
        configuration dictionary including e.g. what function and window to use
    return_full : bool
        return the full fit dictionary


    Returns
    -------
    list
        fit results, including NaN/none for failed lines
        By default, only the centroids, but optionally the full array of outputs.
    """

    # set up parameters for fits
    fit_width = Config['fitwidth']
    fit_function = Config['fitfunction']
    #fit_discretize = Config['discretize'] # NOT USING THIS OPTION PRESENTLY
    n_pixels = Config['n_pixels']
    if Config['use_variance']:
        if variance is not None:
            sigma = np.sqrt(variance)
        else:
            logging.warning('Linefits is configured to use the variances, but no variance was provided. Reverting to no variance.')
            sigma = None

    #xx_pix = np.arange(n_pixels)
    assert len(xx) == len(flux)

    out = fitLib.fit_lines_order(xx_pix,flux,window_centers,sigma=sigma,fitfunction=fit_function,
                                     fit_width_pix=fit_width,basic_window_check=True)

    assert len(window_centers) == len(out)

    if return_full:
        return(out)

    out_centroids = np.full(len(window_centers), np.NaN)

    for mi in range(len(out)):
        out_centroids[mi] = out[mi]['centroid_pix']

    return(out_centroids)

def pool_measure_and_save_linefits(cmdset):
    # override outdir if in cmdset
    #if 'outdir' in cmdset:
    #    outdir = cmdset['outdir']
    #else:
    #    outdir = None # otherwise use value in Config
    out = measure_and_save_linefits(cmdset['filename'],
                                               cmdset['fiber'],
                                               cmdset['Config']) #deleted outdir = outdir


def measure_and_save_linefits(filename,fiber,Config): #deleted outdir = None
    # set up parameters for fits
    fit_width = Config['fitwidth']
    fit_function = Config['fitfunction']
    #fit_discretize = Config['discretize'] # NOT USING THIS OPTION PRESENTLY
    #n_pixels find from wavecal below
    source = Config['source']

    peak_path = Config['master_peak_locs']
    initial_peak_locs_pix = np.load(peak_path,allow_pickle=True,encoding='latin1')[()]
    # this should be a nested ordered dictionary, where the first level is indexed by the orders of the spectrum to analyze
    # and the next level is the mode index number within each order. (this is arbitrary and just for labeling)

    master_wcal_path = Config['master_wavecal']
    #master_wcal = fits.getdata(master_wcal_path)
    master_wcal = fits.open(master_wcal_path).copy()

    n_pixels = np.shape(master_wcal[1].data)[1]
    
    xx_pix = np.arange(n_pixels)

    logging.info('Deriving {} parameters for {} spectrum in {}'.format(fiber,source,filename))

    if not os.path.isfile(filename):
        logging.warning('No file, skipping: {}'.format(filename))
        return

    logging.info('  ... Loading "{}"'.format(filename))

    dataObj = fits.open(filename).copy()

    out_all = OrderedDict()
    out_all_slim = OrderedDict()
    
    for order in initial_peak_locs_pix.keys():
        flux = fitLib.getData(dataObj,fiber,'flux')[order,:] #getattr(dataObj_this,fluxname).data[order,:]
        var = fitLib.getData(dataObj,fiber,'var')[order,:] #getattr(dataObj_this,varname).data[order,:]
        if Config['wavecalmode'] == 'master':
            wave = fitLib.getData(master_wcal,fiber,'wave')[order,:]
        elif Config['wavecalmode'] == 'current':
            try:
                wave = fitLib.getData(dataObj,fiber,'wave')[order,:]
            except NameError:
                logging.warning('Problem with wave ext for file {}, reverting to master'.format(filename))
                wave = fitLib.getData(master_wcal,fiber,'wave')[order,:]
        else:
            logging.warning('wavecal mode unrecognized, defaulting to master')
            wave = fitLib.getData(master_wcal,fiber,'wave')[order,:]

        if Config['subtractcontinuum']:
            fl_subtracted, _, _ = fitLib.subtract_Continuum_fromlines(flux)
            flux = fl_subtracted
        
        if Config['usevariance']:
            sigma = np.sqrt(var)
        else:
            sigma = None
        
        out_all[order] = fitLib.fit_lines_order(xx_pix,flux,initial_peak_locs_pix[order],sigma=sigma,
            fitfunction=fit_function,fit_width_pix=fit_width,basic_window_check=True,wl=wave)
        # if desired to peel out a subset of parameters, could copy over the measure_centroids wrapper

        out_all_slim_order = OrderedDict()

        for mi in out_all[order].keys():
            out_all_slim_order[mi] = OrderedDict()
            out_all_slim_order[mi]['centroid_pix'] = out_all[order][mi]['centroid_pix']
            out_all_slim_order[mi]['centroid_wl'] = out_all[order][mi]['centroid_wl']
            out_all_slim_order[mi]['fwhm_pix'] = out_all[order][mi]['fwhm_pix']
            out_all_slim_order[mi]['fwhm_wl'] = out_all[order][mi]['fwhm_wl']
            out_all_slim_order[mi]['snr_peak'] = out_all[order][mi]['snr_peak']
        out_all_slim[order] = out_all_slim_order

    #if outdir is None:
    outdir = Config['outdir']
    outdir_slim = Config['outdir_slim']    

    if Config['save_full']:
        outname = os.path.join(outdir,os.path.splitext(os.path.basename(filename))[0]+'_'+fiber+'_'+source)
        np.save(outname,out_all)

    if Config['save_slim']:
        outname_slim = os.path.join(outdir_slim,os.path.splitext(os.path.basename(filename))[0]+'_'+fiber+'_'+source+'_slim')
        np.save(outname_slim,out_all_slim)

    return(out_all)


def main(raw_args=None):
    #starttime = datetime.datetime.now()
    args = parse_args(raw_args)    

    if args.logfile is None:
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                            level=logging.getLevelName(args.loglevel))
    else:
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                            level=logging.getLevelName(args.loglevel), 
                            filename=args.logfile, filemode='a')
        install_mp_handler()   #MP Logging!
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) # Sent info to the stdout as well

    #Setup thread pool - do this after multiprocessing logging
    if args.noCPUs > 1:
        logging.info('Building Thread Pool with {} CPUs'.format(args.noCPUs))
        pmap = Pool(int(args.noCPUs)).map  
    else:
        pmap = map    # Do not use Pool.map in single thread mode

    Config = create_configdict_from_file(args.ConfigFile)

    if args.filelist:
        filesToMeasure =  np.genfromtxt(args.TargetFile,dtype=str,ndmin=1)
    else:
        filesToMeasure = [args.TargetFile]
    
    # OrderedDict()
    # fiberkeys = ['CAL-OBJ'] #fiberkeys = ['SCI-OBJ','CAL-OBJ','SKY-OBJ']
    # for fi in fiberkeys:
    #     filesToMeasure[fi] = []

    # if args.filelist is not None:
    #     logging.info(' Using file list {}'.format(args.filelist))
    #     if os.path.exists(args.filelist):
    #         allfiles_in = np.genfromtxt(args.filelist,dtype=str)
    #         allfiles = []
    #         for fi in allfiles_in:
    #             if os.path.exists(fi):
    #                 allfiles.append(fi)
    #             else:
    #                 logging.warning(' ... file: {} does not exist, skipping'.format(fi))
    #     else:
    #         logging.warning(' File list does not exist')
    #         allfiles = []
    #         #logging.warning(' File list {} does not exist, defaulting to Level1Dir from config file')
    #         #allfiles = sorted(glob.glob(os.path.join(Level1Dir,'**/neidL1*.fits')))
    # else:
    #     logging.warning(' Must provide a file list')
    #     allfiles = []
    #     #logging.info(' Using files from Level1Dir {}'.format(Level1Dir))
    #     #allfiles = sorted(glob.glob(os.path.join(Level1Dir,'**/neidL1*.fits')))

    cmdset = []
    fibername = Config['fiber']
    for file in filesToMeasure:
        cmdset.append({'fiber':fibername,'filename':file,'Config':Config}) #deleted 'outdir':outdir

    out_all = list(pmap(pool_measure_and_save_linefits,cmdset))


    #Cleanup threadpool
    try:
        pmap.__self__.close()
    except AttributeError:
        pass  # ignore if pmap was normal map function
    else:
        pmap.__self__.join()

    #veltime = datetime.datetime.now()
    #diagname = InputDate + '_' + sourcekey + '_linefits_diag'
    #diagout = os.path.join(diagdir,diagname)
    #np.save(diagout,diag)

if __name__ == "__main__":
    main()
