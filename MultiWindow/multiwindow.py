#multiwindow
#
#Copyright (C) 2014 Luke Schmidt
#lschmidt@mro.nmt.edu
#
#multiwindow is a module to read in the raw text files output by the 
#"test_multiwindow" program and do something useful with them.
#
#text file format is as follows:
#Frame and Epoch labels
#frame and epoch data values
#Data label
#pixel values of each frame, separated by spaces and output in hex
#
#Each pixel in a line will have 4 values, one for each quadrant.
#consecutive pixels in the first line are read out, each one causing four
#pixel values to arrive, until line readout is complete.  Consecutive 
#lines are read out in the same way.  
#
#L1P1Q1 L1P1Q2 L1P1Q3 L1P1Q4 L1P2Q1 L1P2Q2 L1P2Q3 L1P2Q4 ...
#
#Each pixel can be read out multiple times before moving to the next one
#
#Each line can be read out multiple times before moving on to the next one

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os

#Change these variables to match test_multiwindow.c to ensure proper
#formating of data
READMODE_MULTIPIXEL = 0
READMODE_MULTILINE = 1

#path to output images from --movie option, currently set to current working directory
savepath = os.getcwd() + '/'

#currently only numPixels is used
readTimes = 10 # Number of times to read a pixel or line.
numPixels = 128; # Number of pixels, per line, to be read.
readMode = READMODE_MULTIPIXEL # Multiple reads of pixels (set to 0) or lines (set to 1).

def read(fname):
    """Read in a text file, strip out headers and return epoch and pixel 
    values converted to integers for each frame."""
    mwfile = open(fname, 'r')
    rawmw = mwfile.readlines()
    #nframes is -1 as the last recorded frame has no data
    nframes = (len(rawmw)/4)-1
    print nframes, " frames\n"
    
    #set up arrays to hold pixel data (nframes, pixels, quadrants)
    epochs = np.zeros((nframes))
    line1 = np.zeros((nframes,numPixels*readTimes,4))
    line2 = np.zeros((nframes,numPixels*readTimes,4))
    line3 = np.zeros((nframes,numPixels*readTimes,4))
    line4 = np.zeros((nframes,numPixels*readTimes,4))
    line5 = np.zeros((nframes,numPixels*readTimes,4))
       
   # print frames range is -4 as the last 4 lines of the file has no data
    for i in range(len(rawmw)-4):
        #Extract the epoch
        if i%4 == 1:
            #get rid of extra characters
            frame = rawmw[i].strip('\n').split('\t')
            #extract epoch
            epochs[(i-1)/4] = float(frame[1])
        #Extract the frame data    
        elif i%4 == 3:
            lines = rawmw[i].strip('\n').rstrip().split(' ')
            for l in range(len(lines)):
				 lines[l] = int(lines[l], 16)
            lines = np.array(lines)
            #Split data into individual lines.
            line1[(i-3)/4][:,:] = lines[0:numPixels*readTimes*4].reshape((numPixels*readTimes,4))
            line2[(i-3)/4][:,:] = lines[numPixels*readTimes*4:numPixels*readTimes*8].reshape((numPixels*readTimes,4))
            line3[(i-3)/4][:,:] = lines[numPixels*readTimes*8:numPixels*readTimes*12].reshape((numPixels*readTimes,4))
            line4[(i-3)/4][:,:] = lines[numPixels*readTimes*12:numPixels*readTimes*16].reshape((numPixels*readTimes,4))
            line5[(i-3)/4][:,:] = lines[numPixels*readTimes*16:numPixels*readTimes*20].reshape((numPixels*readTimes,4))
    #return a list of line data  and the timestamps
    return [line1, line2, line3, line4, line5], epochs
    
def plotline_all_quads(mwdata):
    """Display color mapped lines of pixels for a given frame for all 
    quadrants.  fnum from 1-number of frames."""
    
    #frame number
    try:
        fnum=int(raw_input('Input Frame Number:'))
    except ValueError:
        print "Not a number"
    fnum=fnum-1
    
    vmn = 0
    vmx = 30000
    #calculate a reasonable aspect ratio that scales with number of pixels in a line
    asp = 5.0*readTimes/numPixels
    
        
    fig = plt.figure()
    #make a plot for each quadrant
    #quadrant 1
    pltdata = np.zeros((numPixels, readTimes*5))
    
    for r in xrange(readTimes):
        for f in xrange(5):
            pltdata[:,5*r+f] = mwdata[f][fnum,r::readTimes,0]
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(pltdata, interpolation="nearest", vmin=vmn, vmax=vmx, aspect=asp)
    ax1.set_title('Quadrant 1')
        
    #quadrant 2
    pltdata = np.zeros((numPixels, readTimes*5))
    for r in xrange(readTimes):
        for f in xrange(5):
            pltdata[:,5*r+f] = mwdata[f][fnum,r::readTimes,1]
    ax2 = fig.add_subplot(223)
    im2 = ax2.imshow(pltdata, interpolation="nearest", vmin=vmn, vmax=vmx, aspect=asp)
    ax2.set_title('Quadrant 2')
    
    #quadrant 3
    pltdata = np.zeros((numPixels, readTimes*5))
    for r in xrange(readTimes):
        for f in xrange(5):
            pltdata[:,5*r+f] = mwdata[f][fnum,r::readTimes,2]
    ax3 = fig.add_subplot(224)
    im3 = ax3.imshow(pltdata, interpolation="nearest", vmin=vmn, vmax=vmx, aspect=asp)
    ax3.set_title('Quadrant 3')
    
    #quadrant 4
    pltdata = np.zeros((numPixels, readTimes*5))
    for r in xrange(readTimes):
        for f in xrange(5):
            pltdata[:,5*r+f] = mwdata[f][fnum,r::readTimes,3]
    ax4 = fig.add_subplot(222)
    im4 = ax4.imshow(pltdata, interpolation="nearest", vmin=vmn, vmax=vmx, aspect=asp)
    ax4.set_title('Quadrant 4')
    
    fig.suptitle('Frame: {fr}'.format(fr=fnum+1), fontsize=18)
    #make axis for color bar on right side
    cax = fig.add_axes([0.88, 0.1, 0.03, 0.8])
    fig.colorbar(im1, cax=cax)
    
    plt.show()

def plotline_timeseries(mwdata, epoch):
    """For a given line, plot each pixel value over the range of frames.
    Useful for checking intensity changes over the course of data collection."""
    #start at 0 seconds
    time = epoch-epoch[0]
    #get some frame rate information
    frate = (epoch[-1]-epoch[0])/np.shape(epoch)[0]
    hz = 1.0/frate
    print frate, ' seconds per frame, ', hz, ' Hz\n'
    #get quadrant, line and pixel numbers 
    try:
        quadnum=int(raw_input('Input Quadrant Number:'))
    except ValueError:
        print "Not a number"
    try:
        lnum=int(raw_input('Input Line Number:'))
    except ValueError:
        print "Not a number"
    try:
        pixnum=int(raw_input('Input Pixel Number:'))
    except ValueError:
        print "Not a number"    
    
    #average all reads of given pixel
    pltdata = np.average(mwdata[lnum-1][:,pixnum*readTimes:pixnum*readTimes+readTimes,quadnum-1], axis=1)
    
    fig = plt.figure()	
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time, pltdata[:], '.')
    ax.set_ylim([round(pltdata[:].min() - 1000), round(pltdata[:].max() + 1000)])
    #put the title 
    ax.set_title('Quadrant: {quad}, Line: {line}, Pixel: {pix}'.format
                    (quad=str(quadnum), line=str(lnum), pix=str(pixnum)))
    # put the y label somewhere in the middle
    ax.set_ylabel('Count')
    #set x label on last plot
    ax.set_xlabel('Seconds')
    
    plt.tight_layout()
    plt.show()
    
def plotline_movie(mwdata, epoch):
    """output an image sequence of all frames, currently slow, could use
    some optimization, most likely a problem with slow imshow"""
    vmn = 0
    vmx = 30000    
    #calculate a reasonable aspect ratio that scales with number of pixels in a line
    asp = 5.0*readTimes/numPixels
    
    #get number of frames to make into a movie
    try:
        mlen=int(raw_input('Input length of movie in frames: '))
    except ValueError:
        print "Not a number"
    
    #generate figure
    fig = plt.figure()
    txt = fig.suptitle(' ', fontsize=18)
    #set up plot data arrays for each quadrant
    pltdata1 = np.zeros((mlen,numPixels, readTimes*5))
    pltdata2 = np.zeros((mlen,numPixels, readTimes*5))
    pltdata3 = np.zeros((mlen,numPixels, readTimes*5))
    pltdata4 = np.zeros((mlen,numPixels, readTimes*5))
    #fill plot data
    for fnum in range(mlen):
        for r in xrange(readTimes):
            for f in xrange(5):
                pltdata1[fnum,:,5*r+f] = mwdata[f][fnum,r::readTimes,0]
                pltdata2[fnum,:,5*r+f] = mwdata[f][fnum,r::readTimes,1]
                pltdata3[fnum,:,5*r+f] = mwdata[f][fnum,r::readTimes,2]
                pltdata4[fnum,:,5*r+f] = mwdata[f][fnum,r::readTimes,3]
        
    #initialize imshow to speed up plotting        
    initdata = np.zeros((numPixels, readTimes*5))
    
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)
    ax4 = fig.add_subplot(222)
    
    im1=ax1.imshow(initdata, interpolation="nearest", vmin=vmn, vmax=vmx, aspect=asp)
    im2=ax2.imshow(initdata, interpolation="nearest", vmin=vmn, vmax=vmx, aspect=asp)
    im3=ax3.imshow(initdata, interpolation="nearest", vmin=vmn, vmax=vmx, aspect=asp)
    im4=ax4.imshow(initdata, interpolation="nearest", vmin=vmn, vmax=vmx, aspect=asp)
            
    #make a plot for each quadrant
    for fnum in range(mlen):
        fig = plt.gcf()        
        #quadrant 1
        
        im1.set_data(pltdata1[fnum])
        ax1.set_title('Quadrant 1')
        
        #quadrant 2
        
        im2.set_data(pltdata2[fnum])
        ax2.set_title('Quadrant 2')
        
        #quadrant 3
        
        im3.set_data(pltdata3[fnum])
        ax3.set_title('Quadrant 3')
        
        #quadrant 4
        
        im4.set_data(pltdata4[fnum])
        ax4.set_title('Quadrant 4')
        
        txt.set_text('Frame: {fr}'.format(fr=fnum))
        #make axis for color bar on right side
        cax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
        fig.colorbar(im1, cax=cax)
        
        fig.savefig(savepath+str(fnum).zfill(3)+'.png', dpi=72,bbox_inches='tight', pad_inches=0.1)
        

if __name__ == "__main__":
    #Set up command line options
    parser = argparse.ArgumentParser(description="View output generated by test_multiwindow.c")
    parser.add_argument('fname', help='path to file')   
    parser.add_argument("-t", "--timeseries", action="store_true",
                        help="Plot timeseries for a given quadrand, line and pixel.")
    parser.add_argument("-a", "--allquads", action="store_true",
                        help="Plot all lines and quadrants in a given frame as an image.")
    parser.add_argument("-m", "--movie", action="store_true",
                    help="Output a series of png's of each frame, each quadrant shown as an image (SLOW!).")
    args = parser.parse_args()
    
    #Get working file
    mw, epoch = read(args.fname)
    
    #option to plot a single frame of all quadrants
    if args.allquads:
        plotline_all_quads(mw)
    
    #option to plot a timeseries of a single pixel, need line, quadrant and pixel
    elif args.timeseries:
        plotline_timeseries(mw, epoch)
    
    #option to save each plotted frame to a file (same format as plotline_all_quads)
    elif args.movie:
        plotline_movie(mw,epoch)
        
    else:
        print "You didn't choose to do anything, use -h to get a list of options"
    
        
    
