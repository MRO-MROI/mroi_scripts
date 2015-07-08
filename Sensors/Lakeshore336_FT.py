import serial
from serial import SerialException, SerialTimeoutException
import time
from time import gmtime, strftime
import csv
import numpy as np
import sys
#from pylab import *

PROBE  = ['a', 'b', 'c', 'd']
TEMP   = {'a':'', 'b':'', 'c':'', 'd':''}
SAMPLE_RATE = 30 #in seconds

class InstrumentError(Exception):
    """Base class for all exceptions that occur in the instrument

    Attributes:
        msg -- High level explanation of the error.  Should help
                a non-programmer fix the problem.
    """
    def __init__(self, msg):
        self.msg = msg
        print self.msg

class LakeshoreController(object):
    """Represents the Lakeshore 336 Temperature Controller."""
    
    temp_probes = {
        'a':'Radiation Shield',
        'b':'LN2 Tank',
        'c':'C',
        'd':'D',
        }

    def __init__(self):
        """Performs necessary startup procedures."""
        self.__port = '/dev/ttyLakeshore336'
        # Establish a connection to the KMtronic relay board
        self.ser = serial.Serial(port=self.__port, baudrate=57600, bytesize=serial.SEVENBITS,
                                 parity=serial.PARITY_ODD, stopbits=serial.STOPBITS_ONE,
                                 timeout=1)
        """
        Arguments:
            instrument  -- Copy of the NESSI instrument.
            temp_probes -- Dictionary mapping probe character to name.
        Raises:
            InstrumentError

        """
        super(LakeshoreController, self).__init__()

       # self.temp_probes = temp_probes
        self.__port = '/dev/ttyLakeshore336'

        #Open serial connection
        try:
            self.ser    = serial.Serial(port = self.__port, 
                                        baudrate = 57600,
                                        bytesize = serial.SEVENBITS,
                                        parity = serial.PARITY_ODD,
                                        stopbits = serial.STOPBITS_ONE,
                                        timeout=1)
        except ValueError as e:
            msg  = 'Lakeshore Serial call has a programming error in it.\n'
            msg += 'The following ValueError was raised...\n'
            msg += repr(e)
            raise InstrumentError(msg)

        except SerialException as e:
            msg  = 'Lakeshore was unable to connect...\n'
            msg += 'The following SerialException was raised...\n'
            msg += repr(e)
            raise InstrumentError(msg)
            
        except Exception as e:
            raise InstrumentError('An unknown error occurred!\n %s' 
                                  % repr(e))


    def __del__(self):
        """Perform cleanup operations."""
        self.ser.close()
        
    def _completion(self):
        """Generic function that waits to read a completion command."""
    
        while True:
            ch = self.ser.readline()
            if ch != '': 
                break
        return ch

    def _identify(self):
        """Get the current status of the temperature controller"""
        try:
            self.ser.write('*IDN?\n')
        except SerialTimeoutException:
            self.ser.close()
            raise InstrumentError('Writing to Lakeshore controller timed'
                                  ' out. Has it been powered off or '
                                  'disconnected?\n Closed connection to'
                                  ' Lakeshore controller...')
        return self._completion()
    
    def get_temp(self, port):
        """Get the temperature at the given port.
        
        Arguments:
            port -- Character representing which port to read.
                    'a'/'b'/'c'/'d' are the only valid options.

        Raises:
            InstrumentError

        returns String representing temperature at the given port, 
                in kelvin.
        """
        if port not in ['a', 'b', 'c', 'd']:
            raise InstrumentError('A programming error attempted to read'
                                  'a non-existent port on the Lakeshore'
                                  'temperature controller!')
       
        try:
            self.ser.write('KRDG?' + port + '\n')
        except SerialTimeoutException:
            self.ser.close()
            raise InstrumentError('Writing to Lakeshore controller timed'
                                  ' out. Has it been powered off or '
                                  'disconnected?\n Closed connection to'
                                  ' Lakeshore controller...')
        return self._completion()

    def kill(self):
        """Closes connection. Called by a kill_all"""
        self.ser.close()

def open_log():
    #set up file header for log file
    fname = "LS336_" + strftime("%Y%m%d %H:%M:%S", gmtime()) + ".txt"
    with open(fname, "a") as text_file:
        text_file.write('Date, Time, Epoch, Shield, Tank, C, D\n')

    print "Starting Log: " + fname
    return fname

def find_rate(delta, sdelta, tdelta, mdelta, hdelta):
    """Find the current cooling rate for each probe."""
    srate = sdelta/delta
    trate = tdelta/delta
    mrate = mdelta/delta
    hrate = hdelta/delta
    return np.array([srate, trate, mrate, hrate])

def make_plot(fname):
    """Make a new plot every time the temp is read."""
    
    epochs, shieldt, tankt, maskt, H2RGt = [],[],[],[],[]
    with open(fname,'r') as f:
        next(f) # skip headings
        reader=csv.reader(f,delimiter=',')
        for date,epoch,shield,tank,mask,H2RG in reader:
            epochs.append(float(epoch))
            shieldt.append(float(shield))
            tankt.append(float(tank))
            maskt.append(float(mask))
            H2RGt.append(float(H2RG))
    epochs = np.array(epochs)
    shieldt = np.array(shieldt)
    tankt = np.array(tankt)
    maskt = np.array(maskt)
    H2RGt = np.array(H2RGt)
    t = (epochs - epochs[0])/3600. #np.arange(0, len(dates)*SAMPLE_RATE, SAMPLE_RATE)/3600.

    #make the temperature time series plot
    l1 = plot(t, shieldt, 'r-', t, tankt, 'g-', t, maskt, 'b-', t, H2RGt, 'c-')
    legend((l1), ('Shield', 'Tank', 'C', 'D'), 'upper center', shadow=True)
    ylabel('Temperature (K)')
    xlabel('Hours')
    title('Start Time: ' + fname.strip('LakeShoreLog_').strip('.txt'))
    savefig(fname.strip('.txt') + '.png')

    #find the current cooling rate
    #find current and previous values of probes.
    if len(epochs) > 1:
        delta = (epochs[-1] - epochs[-2])/60 # so rate is in K/min
        sdelta = shieldt[-1] - shieldt[-2]
        tdelta = tankt[-1] - tankt[-2]        
        mdelta = maskt[-1] - maskt[-2]
        hdelta = H2RGt[-1] - H2RGt[-2]
        rates = find_rate(delta, sdelta, tdelta, mdelta, hdelta)
    else:
        rates = np.zeros(4)

    #rounds to nearest 30 seconds, used to trigger twitter update
    total_time = round((epochs[-1] - epochs[0])/30.)*30 
    
    return total_time, rates


if __name__=="__main__":
    """Start a log file and record temps."""
    logname = open_log()
    ls = LakeshoreController()
    
    #loop on timer to sample temperatures
    while 1:
        try:
            for p in PROBE:
                TEMP[p] = ls.get_temp(p)

            result = "{time},{epoch},{a},{b},{c},{d}\n".format(time = strftime("%Y%m%d %H:%M:%S", gmtime()), epoch=str(time.time()), a=TEMP['a'].replace('+',''), b=TEMP['b'].replace('+',''), c=TEMP['c'].replace('+',''), d=TEMP['d'].replace('+',''))
            result = result.replace('\n', ' ').replace('\r','')        
            with open(logname, "a") as text_file:
                text_file.write(result + '\n')

            #log_time, rates = make_plot(logname)

            sys.stdout.write('\r')
            sys.stdout.write(' '.ljust(79))
            sys.stdout.write('\r')
            sys.stdout.write(result)
            sys.stdout.flush()

            #sleep for a bit before next temperature check
            time.sleep(SAMPLE_RATE)

        except KeyboardInterrupt:
            print "\n Stopping Temperature Monitoring."
            ls.close()
            break
