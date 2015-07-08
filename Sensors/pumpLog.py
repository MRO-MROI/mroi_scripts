#!/usr/bin/env python

import serial 
import time
import sys
import csv
import numpy as np
#from pylab import *
#from twitter import *

SAMPLE_RATE = 30 #in seconds

OAUTH_TOKEN     = '1590483505-X1yHeRvFpcNoKFVbfHK74LEMdJNp6CwUeWI848v'
OAUTH_SECRET    = 'tZ4x9TK383zovYjAqOI8gxgnHLpYUhyquGC6EOixJeU'
CONSUMER_KEY    = 'nzBepFkOZJVwrgxx00Q'
CONSUMER_SECRET = 'vJ7AqvCkxXH7mZqZaqUGPw1RU7HhMPP4uI3euLTaFZA'

def open_log():
    """set up file header for log file"""
    fname = "TPSCompactLog_" + time.asctime( time.localtime(time.time()) ) + ".txt"
    with open(fname, "a") as text_file:
        text_file.write('Timestamp,Pressure\n')

    print "Starting Log: " + fname
    return fname

def make_plot(name):
    """Make a new plot every time the pressure is read."""
    
    dates, pressures = [],[]
    with open(name,'r') as f:
        next(f) # skip headings
        reader=csv.reader(f,delimiter='\t')
        for date,pressure in reader:
            dates.append(date)
            pressures.append(float(pressure))
            
    t = np.arange(0, len(dates)*SAMPLE_RATE, SAMPLE_RATE)/3600.
    

    #make the plot
    l1 = plot(t, pressures, 'r-')
    ylabel('Torr')
    xlabel('Hours')
    title('Start Time: ' + name[14:].strip('.txt'))
    savefig(name.strip('.txt') + '.png')
    
    total_time = len(dates)*SAMPLE_RATE
    return total_time

def update_twitter(message):
    tw = Twitter(auth=OAuth(OAUTH_TOKEN, OAUTH_SECRET, CONSUMER_KEY, CONSUMER_SECRET))
    tw.statuses.update(status=message)
    print "Updated Twitter at " + time.asctime( time.localtime(time.time()) )

def _read_exit_status(device):
    """Reads exit status from controller.

    Generic function to read from the serial port.
    Returns the first non-empty line

    returns an array of bytes, little-endian.
    """
    while True:
        l = device.readline()
        if l != '': 
            break
    h = l.encode('hex')
    return [h[i] + h[i+1] for i in range(0,len(h)-1,2)], l

def get_pressure(pump): 
    """Poll pump for pressure."""
    send = ''
    pressure = ['\x02','\x80','\x32','\x32','\x34','\x30','\x03','\x38','\x37']
    
    for i in pressure:
        send = send + i
    pump.write(send)
    
    stat = _read_exit_status(pump)
    return float(stat[1][6:13])

def connect_pump():
    """Connect to the pump serial port."""
    x = serial.Serial("/dev/ttyUSB0", baudrate=9600,
                                     bytesize=serial.EIGHTBITS,
                                     parity=serial.PARITY_NONE,
                                     stopbits=serial.STOPBITS_ONE,
                                     timeout=1)
    if x.isOpen():
        print "Connected to Pump \n"
    return x

if __name__=="__main__":
    """Connect to pump and loop on timer to poll pressure."""
    logname = open_log()
    pserial = connect_pump()
    #loop on timer to sample pressure
    while 1:
        try:
            p = get_pressure(pserial)
            sys.stdout.write('\r')
            sys.stdout.write('{0: <50}'.format(p))
            sys.stdout.flush()
            result = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + ','+ str(p)
            with open(logname, "a") as text_file:
                text_file.write(result + '\n')

            #sleep for a bit before next pressure check
            time.sleep(SAMPLE_RATE)

        except KeyboardInterrupt:
            print "\n Stopping Pressure Monitoring."
            break
    pserial.close()
