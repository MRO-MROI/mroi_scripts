import serial
import sys
import time
from time import gmtime, strftime

class LakeshoreController218():
    """Represents the Lakeshore 218 Temperature Controller."""

    def __init__(self):
        """Performs necessary startup procedures."""
        self.ser = serial.Serial(port='/dev/ttyUSB2', 
                                    baudrate=9600, 
                                    bytesize=serial.SEVENBITS,
                                    parity=serial.PARITY_ODD, 
                                    stopbits=serial.STOPBITS_ONE, 
                                    timeout=1)
        if self.ser.isOpen():
            print "Connected to LakeShore 218"
            
    def __del__(self):
        """Perform cleanup operations."""
        self.ser.close()
        
    def ls_comm(self, command):
        self.ser.write(command)
        time.sleep(0.05)
        number_of_bytes = self.ser.inWaiting()
        complete_string = self.ser.readline()
        complete_string = complete_string.replace('\r', '').replace('\n', '').replace('+', '')
        return(complete_string)
        
def open_log():
	"""set up file header for log file"""
	fname = "LS218_" + strftime("%Y%m%d %H:%M:%S", gmtime()) + ".csv"
	with open(fname, "a") as text_file:
		text_file.write('Date Time,Epoch,S1,S2,S3,S4,S5,S6,S7,S8\n')

	print "Starting Log: " + fname
	return fname
	
	
def write_log(logname, msg):
	with open(logname, "a") as text_file:
		text_file.write(msg + '\n')

if __name__ == '__main__':
    ls218 = LakeshoreController218()
    templog = open_log()
    while 1:
		try:
			allports = ls218.ls_comm('KRDG? 0\r\n')

			sys.stdout.write('\r')
			sys.stdout.write(' '.ljust(79))
			sys.stdout.write('\r')
			sys.stdout.write(allports)
			sys.stdout.flush()
			
			result = strftime("%Y%m%d %H:%M:%S", gmtime()) + ',' + str(time.time()) + ',' + allports
			write_log(templog, result)
			
			time.sleep(30)
			
		except KeyboardInterrupt:
			print "\n Stopping LS218 Temperature Monitoring."
			break
