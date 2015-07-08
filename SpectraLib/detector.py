"""detector.py is a set of tools used to characterize the performance of a PICNIC FPA.  
Module is also intended to be used with an IPython Notebook.

****Remember, indexes are switched, x axis is second index in pyfits****

Each metric will be found on a per quadrant basis.
"""

import sys
import glob
import numpy as np
import numpy.ma as ma
from scipy.stats import norm
import matplotlib.pyplot as plt
import pylab as pl
from astropy.io import fits

def readH2RG(fname):
    #assume cds input, does subtraction for you
    hdu = fits.open(fname)
    data0 = hdu[0].data[0]
    data1 = hdu[0].data[1]
    data = data0-data1
    return data

def filelist(listname):
    file = open(listname, 'r')
    names = file.readlines()
    for n in range(len(names)):
        names[n] = names[n].replace('\n', '')
    return names

def deinterlace(hdu):
    frames = hdu[0].header['NAXIS3']
    #shape = (header['NAXIS2'], header['NAXIS1'])
    #data1 = np.zeros((frames, shape[0], shape[1]))
    for f in range(frames):
        hdu[0].data[f] = hdu[0].data[f][:][:,NEWORDER]
        
    return hdu
        
def diffvar(im1, im2):
    diff = im1 - im2
    var = np.var(diff)
    return var  

def summean(im1, im2):
    smean = np.mean(im1) + np.mean(im2)
    return smean

def main():
    #get list of files
    files = filelist("files")
    #set up plotting
    
    # run through each channel
    for ch in range(32):
        means = np.zeros(len(files)/2)
        varis = np.zeros(len(files)/2)
        #process pairs of images
        for i in range(0, len(files), 2):
            im1 = readfile(files[i])[:,ch*64:ch*64+64]
            im2 = readfile(files[i+1])[:,ch*64:ch*64+64]
            print "processing: ", files[i], files[i+1]
            #get sum of means
            som = summean(im1, im2)
            #get variance of difference)
            vod = diffvar(im1, im2)
            means[i/2] = som
            varis[i/2] = vod

        print 'means:', means
        print 'variances: ', varis
        fit = pl.polyfit(varis, means, 1)
        fit_fn = pl.poly1d(fit)
        print "e-/ADU = ", fit[0]
        plt.plot(varis, means, 'o', varis, fit_fn(varis), '--')
        plt.legend(['Channel '+str(ch), str(round(fit[0],3))+' e-/ADU'], loc=9)
        plt.xlabel('Variance')
        plt.ylabel('Mean')
        plt.title('Conversion Gain')
        plt.savefig('ch'+str(ch).zfill(2)+'.png', bbox_inches='tight')
        plt.clf()
    return 0

if __name__ == '__main__':
    main()


def readPICNIC(fname):
    """Read in PICNIC fits file for analysis. Typical file will have some number
    of frames, each frame consisting of the initial CDS read and a DATA read 
    separated by the integration time.  

    Args:
        fname (str) : path to .fits file to be opened

    Returns:
        time (numpy array), data (numpy array) : time is UTC, data will be the 
        difference between the DATA and CDS reads.  data is shape (frames, 256,256)

    Example:
        imtime, imdata = detector.readPICNIC('/home/user/data/image.fits')

    """
    hdu = fits.open(fname)
    imtime = hdu[1].data.field('UTC-READ')
    imdata = hdu[1].data.field('READ1') - hdu[1].data.field('CDS1')
    return imtime, imdata

def splitQuads(frame):
    """frame is split into four quadrants so that each output channel is analyzed 
    separately.

    Args: 
        frame (numpy array): input frame

    Returns:
        q1, q2, q3, q4 (numpy arrays): arrays of (128,128) quadrants counter clockwise
        starting in the upper left."""
    q1 = np.zeros((128,128))
    q2 = np.zeros((128,128))
    q3 = np.zeros((128,128))
    q4 = np.zeros((128,128))

    q2 = frame[0:128,0:128]
    q1 = frame[128:256,0:128]
    q4 = frame[128:256,128:256]
    q3 = frame[0:128,128:256]
    return q1, q2, q3, q4

def plotQuads(q1,q2,q3,q4):
    """Plot each quadrant in same arrangement as full frame.

    Args:
        q1,q2,q3,q4 (numpy arrays): each quadrant

    Returns:
        None
    """
    
    #find max value in all 4 quads so that all on same scale
    upper = 0.0
    for q in [q1,q2,q3,q4]:
        maxval = np.max(q)
        if maxval> upper:
            upper = maxval

    cmap = pl.cm.get_cmap("spectral", int(maxval/100))

    fig = plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.imshow(q1, interpolation="nearest", cmap=cmap, vmax=upper)
    plt.title('Quadrant 1')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.subplot(223)
    plt.imshow(q2, interpolation="nearest", cmap=cmap, vmax=upper)
    plt.title('Quadrant 2')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.subplot(224)
    plt.imshow(q3, interpolation="nearest", cmap=cmap, vmax=upper)
    plt.title('Quadrant 3')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.subplot(222)
    plt.imshow(q4, interpolation="nearest", cmap=cmap, vmax=upper)
    plt.title('Quadrant 4')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()


def plotFrame(frame):
    """Plot a full frame.

    Args:
        frame (numpy array): full frame (256,256)

    Returns:
        None
    """
    #find max value to scale plot
    maxval = np.max(frame)

    fig = plt.figure(figsize=(15,15))
    cmap = pl.cm.get_cmap("spectral", int(maxval/100))
    img = plt.imshow(frame, interpolation="nearest", cmap=cmap, vmin=0, vmax=maxval)
    plt.title('Frame')
    plt.colorbar(spacing='uniform')
    plt.gca().invert_yaxis()
    plt.show()

def plotHist(frame, title='Histogram', xlbl='Value', log=True, line='NaN'):
    """Plot a histogram of given frame."""
    fig = plt.figure(figsize=(10,10))

    n, bins, patches = plt.hist(frame.flatten(), 150, histtype='step')
    if log:
        plt.yscale('log', nonposy='clip')
    plt.xlim([bins[0]-1, bins[-1]+1])
    plt.xlabel(xlbl)
    plt.ylabel('Count')
    if line != 'NaN':
        plt.axvline(line, color='b', linestyle='dashed', linewidth=1)
        plt.axvline(-line, color='b', linestyle='dashed', linewidth=1)
    plt.title(title)


def pixelMask(dark1, time1, dark2, time2):
    """Determine what pixels should be ignored (either non responsive, or hot pixels)
    during further analysis.  Normalize two darks by their exposure time and divide one 
    by the other. (40s dark/40)/(20s dark/20).  Linear pixels will = unity.

    Also plots a histogram and prints mean, median, mode and stddev of result.

    Args:
        dark1, dark2 (numpy array): input frames to be analyzed.
        time1, time2 (float): exposure times of dark1 and dark2

    Returns:
        mask (numpy array): array that is the same size as frame, pixel values are 1, or 0
        1 for good pixel, 0 for bad pixel. Can be plotted later if desired.

    Example:
        mask = detector.pixelMask(imdata)

    """
    mask = np.nan_to_num((dark1/time1)/(dark2/time2))
    plotFrame(mask) 
    plotHist(mask, title='Pixel Mask', log=True)
    return mask


def readNoise(dark1, dark2, cgain):
    """Find the difference of two darks of same exposure time, divide by sqrt(2), apply mean 
    conversion gain.  A histogram will be plotted. 

    Take data with block filter and cover plate installed.

    Args:
        dark1, dark2 (numpy array): input darks to be measured.
        cgain (float): mean conversion gain as previously measured

    Returns:
        noise (numpy array): read noise in e-, can be used to look for patterns."""
    read_noise_frame = cgain*(dark2 - dark1)/np.sqrt(2.0)
    mu = np.mean(read_noise_frame)
    sigma = np.std(read_noise_frame)
    plotHist(read_noise_frame, title='Read Noise', xlbl='e-', log=True)
    print 'Read Noise: ', sigma, 'e-'

def conversionGainFrame(imglist, quadrant):
    """Determine the conversion gain for a given quadrant.  A spatial analysis is done, 
    using all pixels in a quadrant. 

    The mean-variance difference method is used, i.e. the differences between four pairs 
    of flat images are used, each pairs exposure time is twice that of the previous pair.

    Variance = 2*Nread**2 + Signal1 + Signal2

    Will plot both linear and log plots including mean sum vs. variance difference, 
    mean sum vs variance difference - readnoise**2 and a fit to slop for gain in e-/ADU

    see http://www.noao.edu/kpno/manuals/whirc/WHIRC_VI_Mean-variance_101026.pdf
    by Dick Joyce, 2010

    Args:
        imglist (list): File names of image pairs of increasing exposure time
        quadrant (int): Quandrant being analyzed.

    Returns:
        gain, readnoise (float): Gain in e-/ADU and readnoise in e-

    Example:
        g, rn = detector.conversionGainFrame([q1_0s, q1_0s, q1_2s, q1_2s, q1_4s, q1_4s], 3)

    """
    # run through each channel
    means = np.zeros(len(imglist)/2)
    varis = np.zeros(len(imglist)/2)
    #process pairs of images
    for i in range(0, len(imglist), 2):
        im1 = imglist[i]
        im2 = imglist[i+1]
        #get sum of means
        som = summean(im1, im2)
        #get variance of difference)
        vod = diffvar(im1, im2)
        means[i/2] = som
        varis[i/2] = vod

    print 'means:', means
    print 'variances: ', varis
    fit = pl.polyfit(means, varis, 1)
    print fit
    fit_fn = pl.poly1d(fit)
    print 'y intercept', fit_fn(0.0)
    eADU = 1.0/fit[0]
    print "e-/ADU = {0}".format(str(eADU)) 
    fig1 = plt.figure(figsize=(10,10))
    plt.loglog(means, varis, 'o', means, fit_fn(means), '--')
    plt.legend(['Channel '+str(quadrant), str(round(eADU,3))+' e-/ADU'], loc=9)
    plt.xlabel('Mean')
    plt.ylabel('Variance')
    plt.title('Conversion Gain, Quadrant {0}'.format(str(quadrant)))
    plt.show()

    fig2 = plt.figure(figsize=(10,10))
    plt.plot(means, varis, 'o', means, fit_fn(means), '--')
    plt.legend(['Channel '+str(quadrant), str(round(eADU,3))+' e-/ADU'], loc=9)
    plt.xlabel('Mean')
    plt.ylabel('Variance')
    plt.title('Conversion Gain, Quadrant {0}'.format(str(quadrant)))
    plt.show()

