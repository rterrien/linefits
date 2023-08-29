from __future__ import print_function, division, unicode_literals
import numpy as np
import copy
import scipy.optimize
import skimage
from skimage import filters
from skimage import morphology
from scipy import interpolate
from astropy.stats import biweight_location, mad_std
from collections import OrderedDict
import scipy.constants
import logging
import datetime
import astropy
import astropy.time
from packaging import version

""" This is a library of functions that are called in the
wavelength calibration.

"""


def fgauss(x, center, sigma, amp):
    """A Gaussian function.

    This is a standard Gaussian function.

    Parameters
    ----------
    x : float or ndarray of float
        Independent variable for the Gaussian
    center : float or ndarray of float
        Mean for the Gaussian
    sigma : float or ndarray of float
        Standard deviation (sigma) for the Gaussian
    amp : float or ndarray of float
        Amplitude of the Gaussian
    """
    center = float(center)
    sigma = float(sigma)
    amp = float(amp)
    return(amp * np.exp(-((x - center) / sigma) ** 2.))


def fgauss_const(x, center, sigma, amp, offset):
    """Gaussian + offset function.

    This is a Gaussian with a constant offset.

    Parameters
    ----------
    x : float or ndarray of float
        Independent variable for the Gaussian
    center : float or ndarray of float
        Mean for the Gaussian
    sigma : float or ndarray of float
        Standard deviation (sigma) for the Gaussian
    amp : float or ndarray of float
        Amplitude of the Gaussian
    offset : float or ndarray of float
        Offset for the Gaussian
    """
    center = float(center)
    sigma = float(sigma)
    amp = float(amp)
    offset = float(offset)
    return(float(amp) * np.exp(-((x - center) / sigma) ** 2.) + offset)


def fgauss_line(x, center, sigma, amp, offset, slope):
    """Gaussian + line function.

    This is a Gaussian with a linear offset.

    Parameters
    ----------
    x : float or ndarray of float
        Independent variable for the Gaussian
    center : float
        Mean for the Gaussian
    sigma : float or ndarray of float
        Standard deviation (sigma) for the Gaussian
    amp : float or ndarray of float
        Amplitude of the Gaussian
    offset : float or ndarray of float
        Offset for the Gaussian linear offset (y-intercept)
    slope : float or ndarray of float
        Slope for the Gaussian linear offset
    """
    center = float(center)
    sigma = float(sigma)
    amp = float(amp)
    offset = float(offset)
    slope = float(slope)
    return(float(amp) * np.exp(-((x - center) / sigma) ** 2.) + offset + x * slope)

def fgauss_from_1(x, center, sigma, amp):
    """Gaussian + offset function.

    This is a Gaussian with a fixed offset (1 = continuum). Convenience function.

    Parameters
    ----------
    x : float or ndarray of float
        Independent variable for the Gaussian
    center : float or ndarray of float
        Mean for the Gaussian
    sigma : float or ndarray of float
        Standard deviation (sigma) for the Gaussian
    amp : float or ndarray of float
        Amplitude of the Gaussian (negative to )
    """
    center = float(center)
    sigma = float(sigma)
    amp = float(amp)
    offset = 1.
    return(float(amp) * np.exp(-((x - center) / sigma) ** 2.) + offset)


def discretize_oversample(func, x, *args, **kwargs):
    """Upsample a function.

    This function enables discrete "upsampling" of a function
    by an arbitrary (integer) factor.

    Parameters
    ----------
    func : function
        The mathematical function to be upsampled. First argument must be
        independent variable.
    x : float or ndarray of float
        Independent variable for the function prior to upsampling.
    *args : list
        Arguments to be passed to the mathematical function.
    **kwargs : dict
        Keywords for the upsampling. Includes "factor" as upsampling factor.
    """
    if 'factor' in kwargs.keys():
        factor = kwargs['factor']
    else:
        factor = 10
    assert factor > 1
    x1 = np.amin(x)
    x2 = np.amax(x)
    xx = np.arange(x1 - 0.5 * (1 - 1 / factor),
               x2 + 0.5 * (1 + 1 / factor), 1. / factor) + 0.5 / factor
    values = func(xx, *args)

    values = np.reshape(values, (xx.size // factor, factor))
    return(values.sum(axis=1) * 1. / factor)


def dfgauss(x, *args, **kwargs):
    """Upsampled Gaussian function.

    This is an "upsampled" Gaussian, implemented for convenience of testing
    fit sensitivity to choices about how to discretize the line function.

    Parameters
    ----------
    x : float or ndarray of float
        Independent variable for Gaussian
    *args : list
        Arguments for gaussian, in order: mean, sigma, amplitude
    **kwargs : dict
        Keywords for discretization, including: factor
    """
    return(discretize_oversample(fgauss, x, *args, **kwargs))


def dfgauss_const(x, *args, **kwargs):
    """Upsampled Gaussian+constant function.

    This is an "upsampled" Gaussian + constant, implemented for convenience of testing
    fit sensitivity to choices about how to discretize the line function.

    Parameters
    ----------
    x : float or ndarray of float
        Independent variable for Gaussian
    *args : list
        Arguments for gaussian, in order: mean, sigma, amplitude, offset
    **kwargs : dict
        Keywords for discretization, including: factor
    """
    return(discretize_oversample(fgauss_const, x, *args, **kwargs))


def dfgauss_line(x, *args, **kwargs):
    """Upsampled Gaussian+line function.

    This is an "upsampled" Gaussian + line, implemented for convenience of testing
    fit sensitivity to choices about how to discretize the line function.

    Parameters
    ----------
    x : float or ndarray of float
        Independent variable for Gaussian
    *args : list
        Arguments for gaussian, in order: mean, sigma, amplitude, offset
    **kwargs : dict
        Keywords for discretization, including: factor
    """
    return(discretize_oversample(fgauss_line, x, *args, **kwargs))



def rescale(x, oldmin, oldmax, newmin, newmax):
    """Linearly rescale and offset a series.

    This function takes a series and linearly scales/offsets
    it to a new domain. Useful for calling Legendre polynomial, e.g.

    Parameters
    ----------
    x : ndarray
        Original series, float
    oldmin : float
        min of old domain
    oldmax : float
        max of old domain
    newmin : float
        min of new domain
    newmax : float
        max of new domain

    Returns
    -------
    Standardized series.
    """
    out = (newmax - newmin) / (oldmax - oldmin) * (x - oldmin) + newmin
    return(out)



def fitProfile(inp_x, inp_y, fit_center_in, fit_width=8, sigma=None,
               func='fgauss_const', return_residuals=False,p0=None,bounds=(-np.inf,np.inf)):
    """Perform a least-squares fit to a peak-like function.

    Parameters
    ----------
    inp_x : ndarray of float
        x-values of line to be fit (full array; subset is
        taken based on fit width)
    inp_y : ndarray of float
        y-values of line to be fit (full array; subset is
        taken based on fit width)
    fit_center_in : float
        Index value of estimated location of line center;
        used to select region for fitting
    fit_width : int, optional
        Half-width of fitting window. (the default is 8)
    sigma : ndarray of float, optional
        The standard error for each x/y value in the fit.
        (the default is None, which implies an unweighted fit)
    func : {'fgauss','fgauss_const','fgauss_line','fgauss_from_1'} , optional
        The function to use for the fit. (the default is 'fgauss')
    return_residuals : bool, optional
        Output the fit residuals (the default is False)
    p0 : list of first-guess coefficients. The fit can be quite sensitive to these
        choices.
    bounds : Directly sent to scipy.optimize.curve_fit()


    Returns
    -------
    dict of fit parameters:
        {'centroid': fitted centroid
        'e_centroid': std error of fitted peak centroid (covar diagonals)
        'sigma': fitted sigma of peak
        'e_sigma': std error of fitted sigma of peak (covar diagonals)
        'nanflag': are there NaNs present
        'pcov': covariance array - direct output of optimize.curve_fit
        'popt': parameter array - direct output of optimize.curve_fit
        'function_used': function used for fitting
        'tot_counts_in_line': simple sum of y-values in used line region
        'fit_successful': bool, did the fit give a non-errored output?
        'scale_value': scaling factor used to normalize y-values
        'residuals': optional, differences btwn data and optimized model output}
    """

    # select out the region to fit
    # this will be only consistent to +- integer pixels
    fit_center = copy.copy(fit_center_in)
    xx_index = np.arange(len(inp_x))
    assert len(inp_x) == len(inp_y)
    
    j1 = int(np.round(np.amax([0, fit_center - fit_width])))
    j2 = int(round(np.amin([np.amax(xx_index), fit_center + fit_width])))

    # define sub-arrays to fit
    sub_x1 = inp_x[j1:j2]
    sub_y1 = inp_y[j1:j2]

    tot_counts_in_line = float(np.nansum(sub_y1))

    # normalize the sub-array
    try:
        scale_value = np.nanmax(sub_y1)
    except ValueError as e:
        print(e,j1,j2,sub_x1,sub_y1)
    sub_y_norm1 = sub_y1 / scale_value

    # select out the finite elements
    ii_good = np.isfinite(sub_y_norm1)
    sub_x = sub_x1[ii_good]
    sub_y_norm = sub_y_norm1[ii_good]
    if sigma is not None:
        sub_sigma1 = sigma[j1:j2]
        ii_good = np.isfinite(sub_y_norm1) & (np.isfinite(sub_sigma1))
        sub_sigma = sub_sigma1[ii_good]
        sub_y_norm = sub_y_norm1[ii_good]
    else:
        sub_sigma = None

    # note whether any NaNs were present
    if len(sub_x) == len(sub_x1):
        nanflag = False
    else:
        nanflag = True

    # set up initial parameter guesses, function names, and bounds. 
    # initial guess assumes that the gaussian is centered at the middle of the input array
    # the sigma is "1" in x units
    # the amplitude is -0.1.
    # for the functions with an additional constant and line, the constant defaults to 1.
    if func == 'fgauss':
        if p0 is None:
            p0 = (np.mean(sub_x), 5., -0.5)
        use_function = fgauss
    elif func == 'fgauss_const':
        if p0 is None:
            p0 = (np.mean(sub_x),1., -np.ptp(sub_y_norm), np.nanmedian(sub_y_norm))
        use_function = fgauss_const
    elif func == 'fgauss_line':
        if p0 is None:
            p0 = (np.mean(sub_x), 1., -0.5, 1., 0.)
        use_function = fgauss_line
    elif func == 'fgauss_from_1':
        if p0 is None:
            p0 = (np.mean(sub_x),1., -np.ptp(sub_y_norm))
        use_function = fgauss_from_1
    else:
        raise ValueError

    # perform the least squares fit
    try:
        popt, pcov = scipy.optimize.curve_fit(use_function,
                                              sub_x,
                                              sub_y_norm,
                                              p0=p0,
                                              sigma=sub_sigma,
                                              maxfev=10000,
                                              bounds=bounds)

        # Pull out fit results
        # fitted values (0 is the centroid, 1 is the sigma, 2 is the amp)
        # lists used to facilitate json recording downstream
        errs = np.diag(pcov)
        centroid = popt[0]
        centroid_error = np.sqrt(errs[0])
        width = popt[1]
        width_error = np.sqrt(errs[1])
        fit_successful = True
        pcov_list = pcov.tolist()
        popt_list = popt.tolist()

    except RuntimeError:
        errs = np.NaN
        centroid = np.NaN
        centroid_error = np.NaN
        width = np.NaN
        width_error = np.NaN
        fit_successful = False
        pcov_list = []
        popt_list = []

    except ValueError as e:
        print('ValueError: {}'.format(e))
        errs = np.NaN
        centroid = np.NaN
        centroid_error = np.NaN
        width = np.NaN
        width_error = np.NaN
        fit_successful = False
        pcov_list = []
        popt_list = []

    except TypeError as e:
        print('TypeError: {}'.format(e))
        errs = np.NaN
        centroid = np.NaN
        centroid_error = np.NaN
        width = np.NaN
        width_error = np.NaN
        fit_successful = False
        pcov_list = []
        popt_list = []

    except:
        print('unknown error')
        errs = np.NaN
        centroid = np.NaN
        centroid_error = np.NaN
        width = np.NaN
        width_error = np.NaN
        fit_successful = False
        pcov_list = []
        popt_list = []


    if np.isnan(centroid_error) or np.isnan(centroid):
        fit_successful = False

    # build the returned dictionary
    retval = {'centroid': centroid,
              'e_centroid': centroid_error,
              'sigma': width,
              'e_sigma': width_error,
              'nanflag': nanflag,
              'pcov': pcov_list,
              'popt': popt_list,
              'indices_used': (j1, j2),
              'function_used': func,
              'tot_counts_in_line': tot_counts_in_line,
              'fit_successful': fit_successful,
              'scale_value':float(scale_value)}

    # since residual array can be large, optionally include it
    if return_residuals:
        if fit_successful:
            predicted = use_function(sub_x, *popt)
            residuals = (predicted - sub_y_norm).tolist()
        else:
            residuals = np.NaN
        retval['residuals'] = residuals

    #return(retval['popt'][0], retval['popt'][1], retval['popt'][2], retval)
    return(retval)



def bugfix_biweight_location(array,**kargs):
    """ Temperory bug fix for biweight_location which returns nan for zero varience array """
    array = array[~np.isnan(array)] # Remove any nans
    if np.any(mad_std(array,**kargs)==0):
        return np.median(array,**kargs)
    else:
        return biweight_location(array,**kargs)

def subtract_Continuum_fromlines(inputspec,refspec=None,thresh_mask=None,thresh_window=21,mask_dilation=2,spline_kind='cubic'):
    """ Returns a smooth continuum subtracted `inputspec` . If `refspec` is provided, it is used to create the mask fo the continuum region.
    """ 
    # Use inputspec for thersholding if refspec is not provided
    if refspec is None:
        refspec = inputspec

    Xaxis = np.arange(len(refspec))

    if thresh_mask is None:
        # Create a mask for the emission lines
        ThresholdMask = np.atleast_2d(refspec) > filters.threshold_local(np.atleast_2d(refspec), thresh_window,offset=0)
        # Dilate the mask
        if version.parse(skimage.__version__) < version.parse('0.19'):
            ThresholdMask = morphology.binary_dilation(ThresholdMask,selem=np.array([[1]*mask_dilation+[1]+[1]*mask_dilation]))[0]
        else:
            ThresholdMask = morphology.binary_dilation(ThresholdMask,footprint=np.array([[1]*mask_dilation+[1]+[1]*mask_dilation]))[0]
    else:
        ThresholdMask = thresh_mask

    pix_pos_list = []
    continuum_list = []
    for sli in np.ma.clump_unmasked(np.ma.array(refspec,mask=ThresholdMask)):
        pix_pos_list.append(np.mean(Xaxis[sli]))
        continuum_list.append(bugfix_biweight_location(inputspec[sli]))

    Continuum_Func = interpolate.interp1d(pix_pos_list,continuum_list,kind=spline_kind,fill_value='extrapolate')
    Continuum = Continuum_Func(Xaxis)
    outspec = inputspec - Continuum

    return outspec, Continuum, ThresholdMask


def fit_lines_order(xx,fl,peak_locs,sigma=None,wl=None,pix_to_wvl=None,pix_to_wvl_per_pix=None,fitfunction='fgauss_const',
                    fit_width_pix=8,basic_window_check=True):
    """ Fit all peaks in an order

    This is a wrapper for the fitProfile function to do the (often used) task of repeated fitting of many lines
    in a single spectral order.

    Parameters
    ----------
    xx : ndarray of float
        ndarray of pixel values
    fl : ndarray of float
        ndarray of fluxes
    peak_locs : list or dict
        List of peak locations to center fit windows on.
        If dict, the peaks are labaled by their resspective keys.
        If list, the peaks are given sequential labels.
    sigma : optional, ndarray of float
        ndarray of sigma values to send to fitter
    wl : optional, ndarray of float
        ndarray of wavelength values.
        If not provided, no wavelength values are output.
    pix_to_wvl : optional, function
        Function for translating pixel to wavelength
        If not provided, a cubic spline is used
    pix_to_wvl_per_pix : optional, function
        Function for translating pixel to dispersion
        If not provided, a cubic spline is used
    fitfunction : str, optional
        Name of function to fit to lines, name must be accepted by fitProfile
    fit_width_pix : int
        Half-width of fitting window in pixels
    basic_window_check : bool
        Check whether the fitted centroid falls in given peak_loc +- fit_width_pix
        Return NaNs if not
    
    Returns
    -------
    OrderedDict of fit results. Each entry has (key,value) where
    key = peak label as defined by input dictionary (or sequential labels if not provided)
    value = OrderedDict of fit parameters as given by fitProfile.
        
    .. note::
        The interpolation functions are planned to be upgraded to a more stable form
        (e.g. cumsum or PCHIP based)
    """
    if not isinstance(peak_locs,dict):
        peak_locs_dict = OrderedDict()
        mode_names = range(len(peak_locs))
        for mi in mode_names:
            peak_locs_dict[mi] = peak_locs[mi]
    else:
        peak_locs_dict = copy.deepcopy(peak_locs)

    out = OrderedDict()

    # if we have a wavelength array, also translate the (d)pixels to (d)wavelengths
    if wl is not None:
        if pix_to_wvl is None:
            pix_to_wvl = scipy.interpolate.interp1d(xx,wl,kind='cubic',bounds_error=False)
        if pix_to_wvl_per_pix is None:
            dwl = np.diff(wl)
            dwl = np.append(dwl,dwl[-1])
            pix_to_wvl_per_pix = scipy.interpolate.interp1d(xx,dwl,kind='cubic',bounds_error=False)

    for mi in peak_locs_dict.keys():
        loc_this = peak_locs_dict[mi]
        if fitfunction == 'fgauss_const':
            p0 = [loc_this,2.5,1.,0.]
        elif fitfunction == 'fgauss_line':
            p0 = [loc_this,2.5,1.,0.,0.]
        elif fitfunction == 'fgauss':
            p0 = [loc_this,2.1,1.]
        else:
            raise ValueError
        
        try:
            tmp = fitProfile(xx,fl,loc_this,fit_width=fit_width_pix,sigma=sigma,
                                                    func=fitfunction,p0=p0)
        except (RuntimeError, ValueError, RuntimeWarning) as e:
            tmp = OrderedDict()
            tmp['fit_successful'] = False
            tmp['sigma'] = np.NaN
            tmp['scale_value'] = np.NaN
            tmp['centroid'] = np.NaN

            logging.warning('  ... ...  Raised "{0}"  on mode {1}'.format(e, mi))
        #tmp['centroid_wl'] = interp(tmp['centroid'],xx_pix,xx_test)
        centroid_pix = tmp['centroid']

        if basic_window_check:
            check_val = np.abs(loc_this - float(centroid_pix))
            if check_val > fit_width_pix:
                centroid_pix = np.nan
                tmp['fit_successful'] = False

        if wl is not None:
            dwl_per_pix = pix_to_wvl_per_pix(tmp['centroid'])
            centroid_wl = pix_to_wvl(centroid_pix)[()]
        else:
            dwl_per_pix = np.NaN
            centroid_wl = np.NaN

        fwhm_pix = 2.36 * tmp['sigma']
        fwhm_wl = fwhm_pix * dwl_per_pix
        fwhm_vel = fwhm_wl / centroid_wl * 3e8
        peak_counts = tmp['scale_value']

        if not tmp['fit_successful']:
            fwhm_pix = np.NaN
            fwhm_wl = np.NaN
            fwhm_vel = np.NaN
            peak_counts = np.NaN

        out1 = OrderedDict()
        out1['fit_output'] = tmp
        out1['centroid_pix'] = centroid_pix
        out1['centroid_wl'] = centroid_wl
        out1['fwhm_pix'] = fwhm_pix
        out1['fwhm_wl'] = fwhm_wl
        out1['snr_peak'] = np.sqrt(peak_counts)
        out1['prec_est'] = 0.4 * fwhm_vel / (np.sqrt(fwhm_pix) * np.sqrt(peak_counts))

        #print(mi)
        #if mi == 89:
        #    print(out1)
        out[mi] = out1
    return(out)

def redshift(x, vo=0., ve=0.,def_wlog=False):
    """ 
    
    Doppler shift a wavelength array.
    
    Parameters
    ----------
    x : float or ndarray of float
        The wavelengths to be shifted.
    vo : optional, float
        The velocity of the observer [m/s]. (the default is 0.)
    ve : optional, float
        The velocity of the emitter [m/s]. (the default is 0.)
    def_wlog : bool, optional
        Is the input in logarithmic wavelength? (the default is False)
    
    Returns
    -------
    float or ndarray of float
        The emitted wavelength(s).
    """
    if np.isnan(vo):
        vo = 0     # propagate nan as zero
    a = (1.0+vo/scipy.constants.c) / (1.0+ve/scipy.constants.c)
    if def_wlog:
        return x + np.log(a)   # logarithmic
        #return x + a          # logarithmic + approximation v << c
    else:
        return x * a
        #return x / (1.0-v/c)

def datetime_avg(dates):
    ''' Return the average time for a list of datetime objects
    
    Parameters
    ----------
    dates : list of datetimese
    
    Returns
    -------
    datetime
        Average datetime
    '''
    reference_date = datetime.datetime(1900, 1, 1)
    if len(dates) == 0:
        return None
    else:
        return(reference_date + sum([date - reference_date for date in dates], 
                                    datetime.timedelta()) / len(dates))

def getData(dataObj,fiber,choice,justext=False):
    """ Helper function to retrieve level 1 data from neidData object

    Parameters
    ----------
    neidDataObj : neidData instance
        neidData object to be parsed

    fiber: str {SCI, CAL, or SKY} 
        fiber to be returned

    choice: str {flux, wave, var} 
        type of data array to be returned

    justext: optional, bool 
        only return extension number for fiber. Default is False

    Return
    ------
    register name or data value, depending on justname

    """
    #Check input validity
    if fiber.upper() not in ['SCI','CAL','SKY']:
        raise ValueError('fiber must be SCI / CAL / SKY, not "{}"'.format(fiber))
    if choice.lower() not in ['flux','var','wave']:
        raise ValueError('choice must be flux / var / wave, not "{}"'.format(choice))

    cf = fiber.upper()
    cc = choice.lower()
    if cf == 'SCI' and cc == 'flux':
        ext = 1
    elif cf == 'SCI' and cc == 'var':
        ext = 4
    elif cf == 'SCI' and cc == 'wave':
        ext = 7
    elif cf == 'CAL' and cc == 'flux':
        ext = 3
    elif cf == 'CAL' and cc == 'var':
        ext = 6
    elif cf == 'CAL' and cc == 'wave':
        ext = 9
    elif cf == 'SKY' and cc == 'flux':
        ext = 2
    elif cf == 'SKY' and cc == 'var':
        ext = 5
    elif cf == 'SKY' and cc == 'wave':
        ext = 8
    if justext:
        return(ext)
    else:
        return(dataObj[ext].data)


def pool_measure_velshift(cmdset):
    """ Parallelization wrapper for measure_velshift
    
    This function translates a list of dicts into the appropriate kw/args for 
    the measure_velshift function. This facilitates parallelization with the 
    method adopted in the NEID-DRP.
    
    Parameters
    ----------
    cmdset : list
        Each element is a dctionary with keys "precal", "postcal" and "wcal"
    
    Returns
    -------
    velocity_shift : list of float
        List of velocity shifts corresponding to input line fits.

    """
    out = measure_velshift(cmdset['Precal'],cmdset['Postcal'],cmdset['Wcal'])
    return out

def measure_velshift(fits1,fits2,wave,pix_to_wavl_funcs=None):
    ''' Measure the velocity shift between two collections of mode fits
    
    Assuming a certain wavelength solution, translate the pixel position
    change of each mode into a velocity.
    
    Parameters
    ----------
    fits1 : Centroid dictionary
        indexed as dict[spectrum_index][order_index][mode_index]
        the stored value is just the pixel centroid (for start of night)

    fits2 : Centroid dictionary
        indexed as dict[spectrum_index][order_index][mode_index]
        the stored value is just the pixel centroid (for start of night)

    wave : ndarray
        Wavecal array

    pix_to_wavl_funcs : optional wavelength solution function
        How to translate pixel to wavelength.
        If not provided, a cubic spline is used.

    Returns
    -------
        List of float
            List of velocity differences corresponding to each line fit in the "fits" inputs.

    Notes
    -----
    This function requires that the two fit dictionaries have the same indexing. If order or mode index
    keys are missing, the function will return None.

    '''
    allvals = []
    if pix_to_wavl_funcs is None:
        xx = np.arange(9216)
        pix_to_wavl_funcs = OrderedDict()
        for oi in fits1.keys():
            pix_to_wavl_funcs[oi] = scipy.interpolate.interp1d(xx,wave[oi],kind='cubic',bounds_error=False)
    for oi in fits1.keys():
        if oi not in fits2.keys():
            logging.warning('Order index {} is not shared between the fit sets being compared'.format(oi))
            return(None)
        for mi in fits1[oi].keys():
            if mi not in fits2[oi].keys():
                logging.warning('Order index {}, mode {} is not shared between the fit sets being compared'.format(oi,mi))
                return(None)
            pix_to_wavl = pix_to_wavl_funcs[oi]
            pix1 = fits1[oi][mi]
            pix2 = fits2[oi][mi]
            wavl1 = pix_to_wavl(pix1)
            wavl2 = pix_to_wavl(pix2)
            diff = (wavl2 - wavl1)/wavl1 * 3e8
            allvals.append(diff)
    return(allvals)


def combine_peak_locations(fitlist):
    ''' Combine fit of pixel centroids for etalon or comb
    
    Parameters
    ----------
    fitlist : list of fit dictionaries
        Each fit dictionary is indexed as dict[spectrum_index][order_index][mode_index]
        and the stored value (itself also a dict) must have the 'centroid_pix' key

    Returns
    -------
    velocity_shift : list of float
        List of velocity shifts corresponding to input line fits.

    '''
    nvals = len(fitlist)
    out = OrderedDict() #fitlist[0]
    for oi in fitlist[0].keys():
        out[oi] = OrderedDict()
    for oi in fitlist[0].keys():
        for mi in fitlist[0][oi].keys():
            vals = []
            for si in range(nvals):
                vals.append(fitlist[si][oi][mi]['centroid_pix'])
            out[oi][mi] = astropy.stats.biweight_location(vals,ignore_nan=True)
    return(out)
