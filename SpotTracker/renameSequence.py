# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 10:14:01 2014

@author: lschmidt
"""

import glob, os
from datetime import datetime, timedelta

def rename (dir, pattern, titlePattern):
    min_sec_ms = '01:10.000'
    min_sec_ms = datetime.strptime(min_sec_ms, "%M:%S.%f")
    for pathAndFilename in sorted(glob.iglob(os.path.join(dir, pattern))):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        min_sec_ms_str = min_sec_ms.strftime("%M_%S_%f")[:-3]
        os.rename(pathAndFilename, os.path.join(dir, titlePattern % title + min_sec_ms_str + ext))
        min_sec_ms += timedelta(milliseconds=33)

#[os.rename('/home/lschmidt/Documents/20140820-brs-stability/vibrations/full/'+f, '/home/lschmidt/Documents/20140820-brs-stability/vibrations/full/'+f[:10]+f[-4:]) for f in os.listdir('/home/lschmidt/Documents/20140820-brs-stability/vibrations/full/')]
        
rename (r'/home/lschmidt/Documents/20140902-brs-stability/pipe/', r'*.jpg', r'%s_02_09_2014_11_')
        
        