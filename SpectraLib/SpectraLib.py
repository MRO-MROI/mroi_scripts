from astropy.io import fits
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import pylab
import sys
#import lmfit


def GetBackground(filename):
	hdubg = fits.open(filename)
	bg = (hdubg[1].data.field('READ1') - hdubg[1].data.field('CDS1'))
	return np.mean(bg, axis=0)
	

def GetData(filename, rows, cols, rowoff, coloff, bg):
    hdu = fits.open(filename)    
    data = (np.fliplr(hdu[1].data.field('READ1') - hdu[1].data.field('CDS1') - bg))[:, coloff:cols+coloff, rowoff:rows+rowoff]  # Flip lr so it matches array display
    np.shape(data)
    #commented out as error in plotting if this is required.
    #if ( (rowoff - rows) ==0):
    data = data.reshape(data.shape[0], data.shape[1])  # Discard extra indice
	
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
		#print str(spec['wavelength'][n]) + '\t' + str(spec['phase'][n])
	
	return spec
	

def PixShift(data, pix_shift):
	return np.concatenate([data[pix_shift::,:], data[0:pix_shift,:]], axis=0)		
	

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

def PlotSpec(data, time, nChannels, coloff):
	fig = plt.figure()	
	time = time - time[0]
	for n in range(nChannels):
        #pick out max min values to check dispersion
		minmaxloc = [data[:,n+coloff].argmax(), data[:,n+coloff].argmin()]
		ax = fig.add_subplot(nChannels, 1, n+1)
		ax.plot(time, data[:,n+coloff], 'k')
		ax.plot(time[minmaxloc], data[minmaxloc], 'r|')
		ax.yaxis.set_ticks([round(data[:,n+coloff].min() + 1), round(data[:,n+coloff].max() -1)])
		ax.set_xlim([time.min(), time.max()])
		ax.set_ylim(data.min()-100., data.max()+100.)
		#plt.autoscale(axis='y')

	ax.set_xlabel('sec')	
	#plt.tight_layout()
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


def PlotLogSpec(data, time, logdata, channel):
	offs = np.argmin(abs(logdata['epoch'] - time[0]))
	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.plot(time - time[0], data[:, channel], 'k')
	ax.set_ylabel('ADU')
	ax1 = fig.add_subplot(212)
	ax1.plot(logdata['epoch'][offs:offs+data.shape[0]] - time[0], logdata['offset'][offs:offs+data.shape[0]], 'k') 
	ax1.set_ylabel('meters')
	ax1.set_xlabel('sec')
	return fig

def SpecCheck(beam1, beam2, beam12, bg):
	b1, t1 = GetData(beam1, 3, 10, 0, 0, bg)
	b2, t2 = GetData(beam2, 3, 10, 0, 0, bg)
	b1b2, t1t2 = GetData(beam12, 3, 10, 0, 0, bg)
	b1 = b1.mean(axis=0)
	b2 = b2.mean(axis=0)
	b1b2 = b1b2.mean(axis=0)
	fig = plt.figure()
	ax = fig.add_subplot(131)
	im = ax.imshow(b1, interpolation="nearest")
	ax.set_title('Beam 1')
	fig.colorbar(im)
	ax1 = fig.add_subplot(132)
	im2 = ax1.imshow(b2, interpolation="nearest")
	ax1.set_title('Beam 2')
	fig.colorbar(im2)
	ax2 = fig.add_subplot(133)
	im3 = ax2.imshow(b1b2, interpolation="nearest")
	ax2.set_title('Beam 1+2')
	fig.colorbar(im3)
	return fig

def BeamCheck(beam1, beam2, bg):
	b1 = GetBackground(beam1) - GetBackground(bg)
	b2 = GetBackground(beam2) - GetBackground(bg)
	fig = plt.figure()
	ax = fig.add_subplot(121)
	im = ax.imshow(b1, interpolation="nearest")
	ax.set_title('Beam 1')
	fig.colorbar(im)
	ax1 = fig.add_subplot(122)
	im2 = ax1.imshow(b2, interpolation="nearest")
	ax1.set_title('Beam 2')
	fig.colorbar(im2)
	return fig

def Scan(fits, background):
	#quick script to take raw files and generate plots, etc. for a quick look
	#at a spectral scan.
	
	#get the data
	bg = GetBackground(background)
	data, time = GetData(fits, 1, 10, 3, 0, bg)
	#Plot the data
	PlotSpec(data, time, 10, 0)
	#get visibilities
	vb = Vis(data)
	#get wavelengths
	spec = Spectrum(data, 0.2) # 0.2 is interval, in our case, 0.2mm long scan
	#print everything
	print 'Wavelength \t Phase \t\t Visibility \n'
	for n in range(0,10):
		print "%.4f" % spec['wavelength'][n] + '\t\t' + "%.4f" % spec['phase'][n] + '\t\t' + "%.4f" % vb[n]
	
	
	
		




	

	
	


