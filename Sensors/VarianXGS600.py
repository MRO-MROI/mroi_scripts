import serial
import sys
import time

class Varian(object):
    """Represents the Varian Pressure Guage."""

    def __init__(self):
        """Performs necessary startup procedures."""
        self.ser = serial.Serial(port='/dev/ttyXGS600', 
                                     baudrate=9600, 
                                     bytesize=serial.EIGHTBITS, 
                                     parity=serial.PARITY_NONE, 
                                     stopbits=serial.STOPBITS_ONE, 
                                     timeout=1)
        if self.ser.isOpen():
            print "Connected to Varian XGS600"
            
    def __del__(self):
        """Perform cleanup operations."""
        self.ser.close()
        
    def completion(self):
        """Generic function that waits to read a completion command."""
        ln = ""
        while True:
            ln = self.ser.readline()
            print len(ln), ln
            if ln != '': 
                break
            return ln

    def SensorChangeCheck(self,string):
        num = float(string[1:])
        choice = 0;
        if num < 0.02:
            choice = 1
        return choice

    def IMGPressure(self):
        """Get the current pressure."""
        #self.ser.flushOutput()
        self.ser.write('#0002UIMG1')
        return self.completion() 

    def CNVPressure(self):
        """Get the current pressure."""
        #self.ser.flushOutput()
        self.ser.write('#0002UCNV1') 
        return self.completion() 

if __name__=="__main__":
    vn = Varian()

    while 1:
        try:
            img = vn.IMGPressure()
            cnv = vn.CNVPressure()
            sys.stdout.write('\r')
            sys.stdout.write(' '.ljust(79))
            sys.stdout.write('\r')
            sys.stdout.write('IMG1: ' + str(img) + ' Torr CNV1: ' + str(cnv) + ' Torr ' + str(time.time()))
            sys.stdout.flush()

            time.sleep(3)
        except KeyboardInterrupt:
            print "\n Stopping Pressure Monitoring."
            break
            
    vn.__del__()

