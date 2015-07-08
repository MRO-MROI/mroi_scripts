#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  h2rgDeinterlace.py
#  
#  Copyright 2013 Luke Schmidt, <lschmidt@mro.nmt.edu>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  ****Remember, indexes are switched, x axis is second index in pyfits****
import pyfits as pf
import numpy as np
import glob

NEWORDER = []

for i in range(32):
	NEWORDER = NEWORDER + range(i, 2048, 32)

def readfile(fname):
	hdu = pf.open(fname, do_not_scale_image_data=True, mode='update')
	return hdu

def deinterlace(hdu):
	frames = hdu[0].header['NAXIS3']
	#shape = (header['NAXIS2'], header['NAXIS1'])
	#data1 = np.zeros((frames, shape[0], shape[1]))
	for f in range(frames):
		hdu[0].data[f] = hdu[0].data[f][:][:,NEWORDER]
		
	return hdu
		
def savefile(hdu):
	hdu.flush()	
    

def main():
	files = glob.glob('*.fit')
	for f in files:
		hdu = readfile(f)
		hdu = deinterlace(hdu)
		savefile(hdu)
		print 'deinterlacing ' + f
	return 0

if __name__ == '__main__':
	main()

