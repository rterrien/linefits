# linefits
This is a barebones code to perform repeated least-squares fits of the lines that appear in a spectrograph calibration spectrum. It is essentially a wrapper around scipy.curve_fit. It is set up to work with spectra extracted from the NEID and HPF instrumnets.

## Installation
The package should be pip-installable. Navigate to this directory and:
```
pip install .
```

## Example use
Adjust the config file as desired, including pointing to appropriate reference files. Point the script at a single fits spectrum or a text file listing a series of such spectra as below:
```
python -m linefits.linefits_singleepoch linefits.config [file or list]
```

## Reference files
As seen in the config file:
- master_peak_locs: This should be a npy file with a nested ordered dictionary. The first level is the spectral order (labeled by its index in the fits file), and the second level is the index of the line within that order. The value itself is an estimate of the line location in the 1-d spectrum, in units of pixels. This is used to label and locate the windows for each line fit.
- master_wavecal: This is a fits file of the same form as the spectrum, with accurate wavelengths in the relevant extensions.

LSF Implementation
----

In order to run LSF fitting, set the fit funtion in the config file to 'LSF' This requires additional files that can be found in https://github.com/AshutoshJoshiTIFR/NEIDLSFMODEL 
Specifically, you will need: 
- NEID_LSFMODEL_{fiber you want to use}
- NeidLsf.py and all required local libraries

#### Etalon fitting under 'LSF'
The Etalon does not follow LSF behavior exactly, so instead there is the option to input peak profiles as a dictionary of peaks. If config settings are set to 
>Source = 'Etalon'

and 
>fit_function = 'LSF'

The program will look for a local file containing those peak profiles and attempt to fit them, they must be alligned one-to-one with the indexing used for the peak location file under the config option
>master_peak_locs
