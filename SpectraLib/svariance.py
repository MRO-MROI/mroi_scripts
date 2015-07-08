"""
****Remember, indexes are switched, x axis is second index in pyfits****"""
import pyfits as pf
import numpy as np
import glob
import matplotlib.pyplot as plt
import pylab as pl

def readfile(fname):
	#assume cds input, does subtraction for you
	hdu = pf.open(fname)
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

