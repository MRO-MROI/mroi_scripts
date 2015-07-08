#spottracker
#
#Copyright (C) 2014 Luke Schmidt
#lschmidt@mro.nmt.edu
#
#This script is a generalized tool to read in a series of images, find
#all of the trackable image spots in each image, centroid each spot and
#record it's (x,y) position over time.
#
#


import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import pylab
import sys
import glob
import argparse
from pandas import DataFrame
import pandas as pd
from scipy import misc
import datetime
import calendar
import time
import timeit
import cv2

DEBUG = True

#add path to PyGuide, I think it is some weirdness going on with using
#Anaconda and what paths are set.
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import PyGuide

# Default Parameters
# these settings will need to be updated once we have the real guide camera
# in order, bias (ADU), readnoise (e-), gain (e-/ADU), saturation level (ADU)
ccdInfo = PyGuide.CCDInfo(10.0, 4.01, 1.5, 256)
# these are general settings
thresh = 5.0
radMult = 3.0
rad = 20
satLevel = (2**8)-1
verbosity = 0
doDS9 = False
mask = None
satMask = None


def imageList(fpath):
    """Find all images with a given extention and return their names as
    a list"""
    flist = raw_input("What is the file identifier? (*.?) >")
    images = sorted(glob.glob(fpath+flist))
    if DEBUG:
        for item in images:
            sys.stdout.write('\r'+item)
            sys.stdout.flush()
    print '\n',len(images), " Images found."
    return images
    
def imageToNumpy(image):
    """Take an image found using imageList() and read in the 
    image data to a numpy array."""
    im = misc.imread(image, flatten=True)

    return im
    
def readVideo(vidfile):
    """Read in a video file and extract images."""
    vidreader = cv2.VideoCapture(vidfile)
    count = 0    
    while True:
        gotImage, image = vidreader.read()
        print gotImage, count        
        if gotImage:
            cv2.imwrite(vidfile+"%d.png" %count, image)
            count += 1
        else:
            break    
    
def getTimeDate(imname):
    """Get the time and date from the file name of the image"""
    timedate = imname[-27:-4].split('_')
    timedate = '{yyyy}-{mm}-{dd}T{hh}:{min}:{sec}'.format(yyyy=timedate[2], mm=timedate[1], dd=timedate[0], hh=timedate[3], min=timedate[4], sec=timedate[5], ms=timedate[6])
    timedate = datetime.datetime.strptime(timedate,'%Y-%m-%dT%H:%M:%S')
    utime    = calendar.timegm(timedate.timetuple())+timedate.microsecond/1000000.0

    return timedate,utime

def findSpots(data):
    """Find the initial guess location of all spots to be tracked."""
    stars = PyGuide.findStars(data, mask, satMask, ccdInfo, thresh, radMult, rad, verbosity, doDS9)
    return stars
    
def centroid(data, xy):
    """Find the centroids of all sources in an image."""
    centroid = []
    for i in range(len(xy)):
        centroid.append(PyGuide.Centroid.centroid(data, mask, satMask, xy[0][i].xyCtr, rad, ccdInfo, doSmooth=True))
    return centroid

def open_log():
    loggerfile = "centroids.csv"
    logfile = open(loggerfile, 'a')
    logfile.write('timestamp,centroid_x1,centroid_y1,x1_error,y1_error,centroid_x2,centroid_y2,x2_error,y2_error\n')
    return logfile
    
def write_log(logfile, logdata):
    """Write the centroids, centroid error and timestamps to a text file 
    for future processing."""
    logfile.write(logdata)
    
def plotCentroids(results):
    """Plot the centroids vs. time in x and y."""
    from matplotlib.ticker import NullFormatter
    # the random data
    ti = results[:,8]
    
    x1 = results[:,0]
    y1 = results[:,1]
    
    x2 = results[:,4]
    y2 = results[:,5]
    
    x1err = results[:,2]
    y1err = results[:,3]
    x2err = results[:,6]
    y2err = results[:,7]    

    nullfmt   = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.5
    bottom, height = 0.1, 0.5
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_scatterx = [left, bottom_h, width, 0.35]
    rect_scattery = [left_h, bottom, 0.35, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8,8))
    
    axScatter  = plt.axes(rect_scatter)
    axScatterx = plt.axes(rect_scatterx)
    axScattery = plt.axes(rect_scattery)
    axScatter.grid()

    # no labels
    axScatterx.xaxis.set_major_formatter(nullfmt)
    axScattery.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x1, y1, c=ti, cmap='autumn', linewidth='0.3', s=12)
    axScatter.scatter(x2, y2, c=ti, cmap='winter', linewidth='0.3', s=12)
    axScatter.set_xlabel('Pixel')
    axScatter.set_ylabel('Pixel')
    axScatter.set_ylim(axScatter.get_ylim()[::-1])

    # now determine nice limits by hand:
#    binwidth = 0.25
#    xymax = np.max( [np.max(np.fabs(x1)), np.max(np.fabs(y1))] )
#    lim = ( int(xymax/binwidth) + 1) * binwidth

#    axScatter.set_xlim( (-lim, lim) )
#    axScatter.set_ylim( (-lim, lim) )
    # X and Y time series
    axScatterx.scatter(x1, ti, c=ti, cmap='autumn', linewidth='0.3', s=12)
    axScatterx.scatter(x2, ti, c=ti, cmap='autumn', linewidth='0.3', s=12)
    axScatterx.set_title('X drift')
    axScatterx.set_ylabel('Image')
    
    axScattery.scatter(ti, y1, c=ti, cmap='winter', linewidth='0.3', s=12)
    
    axScattery.scatter(ti, y2, c=ti, cmap='winter', linewidth='0.3', s=12)
    axScattery.set_title('Y drift')
    axScattery.set_xlabel('Image')
    axScattery.set_ylim(axScattery.get_ylim()[::-1])
    
#    axLinex.set_xlim( axScatter.get_xlim() )
#    axLiney.set_ylim( axScatter.get_ylim() )
    plt.savefig('centroid.png', dpi=100,bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
def plotSingleCentroid(results, cnum):
    """Plot a single centroid vs. time in x and y."""
    from matplotlib.ticker import NullFormatter

    # the random data
    ti = results[:,8]
    x1 = results[:,0]
    y1 = results[:,1]
    
    x2 = results[:,4]
    y2 = results[:,5]
    
    x1err = results[:,2]
    y1err = results[:,3]
    x2err = results[:,6]
    y2err = results[:,7]
    
    if cnum == 0:
        x = x1
        y = y1
        xerr = x1err
        yerr = y1err
        color='Reds'
    elif cnum == 1:
        x = x2
        y = y2
        xerr = x2err
        yerr = y2err
        color='Blues'

    nullfmt   = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.5
    bottom, height = 0.1, 0.5
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_scatterx = [left, bottom_h, width, 0.35]
    rect_scattery = [left_h, bottom, 0.35, height]

    # start with a rectangular Figure
    plt.figure(2, figsize=(8,8))

    axScatter  = plt.axes(rect_scatter)
    axScatterx = plt.axes(rect_scatterx)
    axScattery = plt.axes(rect_scattery)
        
    # no labels
    axScatterx.xaxis.set_major_formatter(nullfmt)
    axScattery.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y, c=ti, cmap=color, linewidth='0.3', s=6)
    axScatter.set_xlabel('Pixel')
    axScatter.set_ylabel('Pixel')
    
    axScatter.set_ylim(axScatter.get_ylim()[::-1])
    
    # now determine nice limits by hand:
#    binwidth = 0.25
#    xymax = np.max( [np.max(np.fabs(x1)), np.max(np.fabs(y1))] )
#    lim = ( int(xymax/binwidth) + 1) * binwidth

#    axScatter.set_xlim( (-lim, lim) )
#    axScatter.set_ylim( (-lim, lim) )
    # X and Y time series
    
    #plt.fill(x, y1, 'b', x, y2, 'r', alpha=0.3)
    
    axScatterx.scatter(x, ti, c=ti, cmap=color, linewidth='0.3', s=6)
    axScatterx.plot(x+xerr/2.0, ti, 'k', x-xerr/2.0, ti, 'k',alpha=0.2)
    axScatterx.set_title('X drift')
    axScatterx.set_ylabel('Image')
    
    
    axScattery.scatter(ti, y, c=ti, cmap=color, linewidth='0.3', s=6)
    axScattery.plot(ti, y+yerr/2.0,'k', ti, y-yerr/2.0,'k',alpha=0.2)
    axScattery.set_title('Y drift')
    axScattery.set_xlabel('Image')
    
    #axScattery.set_ylim(axScattery.get_ylim()[::-1])
  
    
    axScatter.set_xlim( axScatterx.get_xlim() )
    axScatter.set_ylim( axScattery.get_ylim() )
    #xlims = axScatter.get_xlim()
    #ylims = axScatter.get_ylim()
    
    axScatter.grid()

    plt.savefig('centroid'+str(cnum)+'.png', dpi=100,bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
def centroidOffset(ti, x1,y1,x2,y2, pixelSize):
    """Plot offset in microns from original positions."""
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    ax0.plot(ti, ((x1-x2)-(x1[0]-x2[0]))*pixelSize, '-')
    ax1.plot(ti, ((y1-y2)-(y1[0]-y2[0]))*pixelSize, 'g-')

    ax0.set_title('Offset From Original Centroid Separation X')
    ax0.set_ylabel('Microns')

    ax1.set_title('Offset From Original Centroid Separation Y')
    ax1.set_xlabel('Image')
    ax1.set_ylabel('Microns')      

    plt.savefig('offset.png', dpi=100,bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
def centroidDistance(ti, x1, y1, x2, y2, pixelSize):
    """Plot the change in straight line distance between two centroids."""
    plt.plot(ti, np.sqrt((x1-x2)**2+(y1-y2)**2)*pixelSize, '-')
    plt.title('Radial Distance Between Centroids')
    plt.xlabel('Image')
    plt.ylabel('Microns')
    plt.savefig('centroidDistance.png', dpi=100,bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    
def centroidOverlay(fname, dat, res):
    """Take the found centroids and overlay them on the original images
    to see how well the centroid algorithm works."""
    implot = plt.imshow(dat)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(6.667, 5)
    x = [res[0].xyCtr[0], res[1].xyCtr[0]]
    y = [res[0].xyCtr[1], res[1].xyCtr[1]]
    xerr = [res[0].xyErr[0], res[1].xyErr[0]]
    yerr = [res[0].xyErr[1], res[1].xyErr[1]]

    plt.errorbar(x, y, xerr, yerr, capsize=0, ls='none', color='w', 
                elinewidth=1)

    plt.savefig(fname[:-37]+'Overlay_'+fname[-37:], dpi=175,bbox_inches='tight', pad_inches=0)
    plt.clf()
    
def extractData(fnames, overlay=False):
    """Convert all images to numpy arrays and match with date and time
    from file name."""
    #open log
    L = open_log()
    #set up plot array
    plotdata = np.zeros((len(fnames),9))
    i=0
    start_time = time.time()
    for f in fnames:
        #get image data
        imdata = imageToNumpy(f)
        #get timestamp
        imtime,utime = getTimeDate(f)
        #get initial centroid guess
        imcentguess = findSpots(imdata)
        #get centroids and centroid errors
        imcentroid = centroid(imdata, imcentguess)
        #overlay found centroid on image and save result
        if overlay:
            centroidOverlay(f,imdata,imcentroid)

        x1=imcentroid[0].xyCtr[0]
        y1=imcentroid[0].xyCtr[1]
        x1e=imcentroid[0].xyErr[0]
        y1e=imcentroid[0].xyErr[1]
        x2=imcentroid[1].xyCtr[0]
        y2=imcentroid[1].xyCtr[1]
        x2e=imcentroid[1].xyErr[0]
        y2e=imcentroid[1].xyErr[1]
        framedat = np.array([x1, y1, x1e, y1e, x2, y2, x2e, y2e, i])
        plotdata[i,:] = framedat
        sys.stdout.write('\rProcessing Frame: '+str(i))
        sys.stdout.flush()
        #write log file
        write_log(L,'{imtime},{x1},{y1},{x1e},{y1e},{x2},{y2},{x1e},{y2e}\n'.format(x1=x1, y1=y1, x1e=x1e, y1e=y1e, x2=x2, y2=y2, x2e=x2e, y2e=y2e, imtime=imtime))
        i+=1
    end_time = time.time()    
    print '\n' + str(int(end_time-start_time)) + ' seconds to process'
    return plotdata
    
def readCSV(infile):
    """Read in CSV file with times, temps, etc."""
    dateparse = lambda x: pd.datetime.strptime(x, '%Y %m %d %H %M %S')
    csvdat = pd.read_csv(infile, parse_dates={'timestamp': ['year','month','day','hour','minute','second']}, date_parser=dateparse, index_col=0)
    
    return csvdat
    
def readLog(infile):
    """Read in csv generated by extractData"""
    logdat = pd.read_csv(infile, parse_dates=[0], index_col=0)
    return logdat
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', help='file path')   
    args = parser.parse_args()
    files = imageList(args.fpath)
    results = extractData(files)
    plotCentroids(results)
    #plot only 1st spot
    plotSingleCentroid(results, 0)
    #plot only second spot
    plotSingleCentroid(results, 1)
    
    
