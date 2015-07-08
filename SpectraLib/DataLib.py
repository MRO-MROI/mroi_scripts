from astropy.io import fits
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import pylab
import sys
import lmfit


def GetBackground(filename):
	hdubg = fits.open(filename)
	bg = (hdubg[1].data.field('READ1') - hdubg[1].data.field('CDS1'))
	return np.mean(bg, axis=0)
	

def GetData(filename, rows, cols, rowoff, coloff, bg):
	hdu = fits.open(filename)
	data = (np.fliplr(hdu[1].data.field('READ1') - hdu[1].data.field('CDS1') - bg))[:, coloff:cols+coloff, rowoff:rows+rowoff]  # Flip lr so it matches array display
	if ( (rowoff - rows) ==0):
		data = data.reshape(data.shape[0], data.shape[1])  # Discard extra indice BUG?
	
	time = hdu[1].data.field('UTC-READ')

	return data, time


def Vis(data):
	return (np.max(data,axis=0) - np.min(data,axis=0)) / (np.max(data,axis=0) + np.min(data,axis=0))


def Spectrum(data, intv): 
	for n in range(data.shape[1]):
		data[:,n] = data[:,n] - np.mean(data[:,n])
	
	freq = np.linspace(0, data.shape[0]-1, data.shape[0]) / data.shape[0] / np.float(intv)
	spec = np.zeros(data.shape[1], dtype={'names': ['wavelength', 'phase'], 'formats': ['f4', 'f4']})
	for n in range(data.shape[1]):
		Data = np.fft.fft(data[:,n])
		mx = np.argmax(abs(Data[0:data.shape[0]/2]))
		spec[n] = (1 / freq[mx], np.angle(Data[mx]))
		print str(spec['wavelength'][n]) + '\t' + str(spec['phase'][n])
	
	return spec	
	

def PeakLocation(data, intv):
	x = np.linspace(0, data.shape[0]-1, data.shape[0]) * intv
	for n in range(data.shape[1]):
		data[:,n] = data[:,n] - np.mean(data[:,n])
		print x[np.argmax(abs(data[:,n]))]

	return x[np.argmax(abs(data), axis=0)] 
	
	
def LogData(filename):
	fid = open(filename)
	headers = fid.readline()
	filedata = fid.readlines()
	
	dtype = {'names': headers.replace('\t\t', '\t').replace('\t\t', '\t').strip('\n').split('\t'),
		'formats': ['float', 'f4', 'f4', 'a5', 'a5', 'f4', 'a5', 'f4', 'f4', 'f4', 'f4', 'f4']}  # Dict for structured array
	data = np.zeros(len(filedata), dtype)
	
	for n in range(len(filedata)):
		temp = filedata[n].replace('\t\t', '\t').strip('\n').split('\t')
		data[n] = (float(temp[0]), float(temp[1]), float(temp[2]), temp[3], temp[4], 
			float(temp[5]), temp[6], float(temp[7]), float(temp[8]), float(temp[9]),
			float(temp[10]), float(temp[11]) )

	return data

def ParseLogVal(val):
        if val in ('true', 'false'):
                return val # or convert to python boolean
        else:
                return float(val)

def NewLogData(filename):
	fid = open(filename)
	headers = fid.readline()
	filedata = fid.readlines()
        names = headers.replace('\t\t', '\t').replace('\t\t', '\t').strip('\n').split('\t')
        startCols = 8
        chanCols = 4
        numChannels = (len(names) - startCols)/chanCols
        formats = ['float', 'f4', 'f4', 'a5', 'a5', 'f4', 'a5', 'f4'] + numChannels*chanCols*['f4']
	data = np.zeros(len(filedata), {'names': names, 'formats': formats})
	for n in range(len(filedata)):
		temp = filedata[n].replace('\t\t', '\t').strip('\n').split('\t')
                values = map(ParseLogVal, temp)
                data[n] = tuple(values)

	return data


def PlotSpec(data, time, nChannels, coloff):
	fig = plt.figure()
	x = np.linspace(0, data.shape[0]-1, data.shape) * intv  # spatial axis
	
	for n in range(nChannels):
		ax = fig.add_subplot(nChannels, 1, n+1)
		ax.plot(time, data[:,n+coloff], 'k')
		ax.yaxis.set_ticks([round(data[:,n+coloff].min() + 1), round(data[:,n+coloff].max() -1)])
		ax.set_xlim([time.min(), time.max()])
		
		#plt.autoscale(axis='y')


	ax.set_xlabel('sec')	
	plt.tight_layout()
	pylab.show()
	return fig
	
	
def PlotSpec2(data, intv, nChannels, coloff, spec):
	fig = plt.figure()
	x = np.linspace(0, data.shape[0]-1, data.shape[0]) * intv  # spatial axis
	
	for n in range(nChannels):
		ax = fig.add_subplot(nChannels, 1, n+1)
		ax1 = ax.twinx()
		ax.plot(x, data[:,n+coloff], 'k')
		ax.yaxis.set_ticks([])
		ax1.yaxis.set_ticks([])
		ax.set_ylabel('%d' % (spec[n]['wavelength']*1000))
		ax1.set_ylabel('%0.2f' % (int(spec[n]['phase']*1000) / 1000.))
		if (n != (nChannels-1)):
			ax.xaxis.set_ticklabels([])
		ax.set_xlim([x.min(), x.max()])
		plt.autoscale(axis='y')


	ax.set_xlabel('microns')	
	plt.tight_layout()
	pylab.show()
	return fig	
	

	
	


