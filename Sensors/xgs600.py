import serial
import time
from time import gmtime, strftime
import sys


class XGS600Driver():
    def __init__(self, port='/dev/ttyUSB3'):
        self.f = serial.Serial(port)

    def xgs_comm(self, command):
        comm = "#00" + command + "\r"

        self.f.write(comm)
        time.sleep(0.25)
        number_of_bytes = self.f.inWaiting()
        complete_string = self.f.read(number_of_bytes)
        complete_string = complete_string.replace('>', '').replace('\r', '')
        return(complete_string)

    def read_all_pressures(self):
        pressure_string = self.xgs_comm("0F")
        #print pressure_string
        if len(pressure_string) > 0:
            temp_pressure = pressure_string.replace(' ', '').split(',')
            pressures = []
            for press in temp_pressure:
                if press == 'OPEN':
                    pressures.append(-1)
                else:
                    try:
                        pressures.append((float)(press))
                    except:
                        pressures.append(-2)
        else:
            pressures = [-3]
        return(pressures)


    def list_all_gauges(self):
        gauge_string = self.xgs_comm("01")
        gauges = ""
        for i in range(0,len(gauge_string),2):
            gauge = gauge_string[i:i+2]
            if gauge == "10":
                gauges = gauges + str(i/2) + ": Hot Filament Gauge\n"
            if gauge == "FE":
                gauges = gauges + str(i/2) + ": Empty Slot\n"
            if gauge == "40":
                gauges = gauges + str(i/2) + ": Convection Board\n"
            if gauge == "3A":
                gauges = gauges + str(i/2) + ": Inverted Magnetron Board\n"
        return(gauges)

    def read_pressure(self, id):
        pressure = self.xgs_comm('02' + id)
        try:
            val = float(pressure)
        except ValueError:
            val = -1.0
        return(val)

    def filament_lit(self, id):
        filament = self.xgs_comm('34' + id) 
        return(int(filament))

    def emission_status(self, id):
        status = self.xgs_comm('32' + id)
        emission = status == '01'
        return emission

    def set_smission_off(self, id):
        self.xgs_comm('30' + id)
        time.sleep(0.1)
        return self.emission_status(id)

    def set_emission_on(self, id, filament):
        if filament == 1:
            command = '31'
        if filament == 2:
            command = '33'
        self.xgs_comm(command + id)
        return self.emission_status(id)

    def read_software_version(self):
        gauge_string = self.xgs_comm("05")
        return(gauge_string)


    def read_pressure_unit(self):
        gauge_string = self.xgs_comm("13")
        unit = gauge_string.replace(' ','')
        if unit == "00":
            unit = "Torr"
        if unit == "01":
            unit = "mBar"
        if unit == "02":
            unit = "Pascal"
        return(unit)

def open_log():
	"""set up file header for log file"""
	fname = "XGS600Log_" + strftime("%Y%m%d %H:%M:%S", gmtime()) + ".csv"
	with open(fname, "a") as text_file:
		text_file.write('Date Time,Epoch,Pressure IMG1,Pressure CNV1\n')

	print "Starting Log: " + fname
	return fname
	
	
def write_log(logname, msg):
	with open(logname, "a") as text_file:
		text_file.write(msg + '\n')

if __name__ == '__main__':
    xgs = XGS600Driver()
    pumplog = open_log()
    unit = xgs.read_pressure_unit()
    while 1:
		try:
			img = str(xgs.read_pressure('UIMG1'))
			cnv = str(xgs.read_pressure('UCNV1'))
			
			sys.stdout.write('\r')
			sys.stdout.write(' '.ljust(79))
			sys.stdout.write('\r')
			sys.stdout.write(img + ' ' + unit + ' ' + cnv + ' ' + unit)
			sys.stdout.flush()
			
			result = strftime("%Y%m%d %H:%M:%S", gmtime()) + ',' + str(time.time()) + ',' + img + ',' + cnv
			write_log(pumplog, result)
			
			time.sleep(30)
			
		except KeyboardInterrupt:
			print "\n Stopping Pressure Monitoring."
			break
