from astropy.io import fits
from math import *
import cmath
import numpy as np

def welch(i, n):
    return (1. - (float(2*i - n + 1) / float(n - 1))**2.)

def triangle(amp, period, t):
    halfp = 0.5*period
    tri = (t+halfp*(0.5-floor(t/halfp+1.0)))*((-1.0)**floor(t/halfp))
    return (amp/halfp*tri)

def get_background(filename):
    hdubg = fits.open(filename)
    bg = (hdubg[1].data.field('READ1') - hdubg[1].data.field('CDS1'))
    return np.mean(bg, axis=0)
	
def get_data(filename, rows, cols, rowoff, coloff, bg):
    hdu = fits.open(filename)
    data = (np.fliplr(hdu[1].data.field('READ1') - hdu[1].data.field('CDS1') - bg))[:, coloff:cols+coloff, rowoff:rows+rowoff]  # Flip lr so it matches array display
    if ( (rowoff - rows) ==0):
        data = data.reshape(data.shape[0], data.shape[1])  # Discard extra indice
	
    time = hdu[1].data.field('UTC-READ')
    return data, time

def parse_log_val(val):
    if val in ('true', 'false'):
        return val # or convert to python boolean
    else:
        return float(val)

def new_log_data(filename):
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
        values = map(parse_log_val, temp)
        data[n] = tuple(values)
    return data


class FTBaselineResults:
    """Class to store fringe engine results for a particular baseline."""

    def __init__(self, numChannels, numGDTrials, numCohFrames):
        self.numChannels = numChannels
        self.numGDTrials = numGDTrials
        self.numCohFrames = numCohFrames
        self.trialGD = np.zeros((numGDTrials,))
        self.groupDelaySpectrum = np.zeros((numGDTrials,), np.complex)
        self.avgGroupDelaySpectrum = np.zeros((numGDTrials,))
        self.estDelay = 0.0
        self.visibility = np.zeros((numChannels,))
        self.phase = np.zeros((numChannels,))
        self.flux = np.zeros((numChannels,))
        self.epoch = 0.0

    def __str__(self):
        str = 'epoch=%lf estDelay=%lf' % (self.epoch, self.estDelay)
        for j in range(self.numChannels):
            str += ' visibility_%d=%lf phase_%d=%lf flux_%d=%lf' % (
                j+1, self.visibility[j], j+1, self.phase[j], j+1, self.flux[j])
        return str

    def asarray(self):
        """Return results as structured array."""
        names = ['epoch', 'estDelay']
        formats = ['float', 'float']
        for j in range(self.numChannels):
            names += ['visibility_%d' % (j+1,), 'phase_%d' % (j+1,),
                      'flux_%d' % (j+1,)]
            formats += ['float', 'float', 'float']
        log = np.zeros(1, {'names':names, 'formats':formats})
        log['epoch'] = self.epoch
        log['estDelay'] = self.estDelay
        for j in range(self.numChannels):
            log['visibility_%d' % (j+1,)] = self.visibility[j]
            log['phase_%d' % (j+1,)] = self.phase[j]
            log['flux_%d' % (j+1,)] = self.flux[j]
        return log


class BasdenEngine:
    """Class for offline processing of ICoNN Closed-Loop Experiment data.
    
    Mimics the real-time processing performed by ftenginegui.

    """
    
    def __init__(self):
        # Configuration parameters
        self.numChannels = 5
        self.numGDTrials = 1000
        # Variable parameters
        self.trialGDScale = 0.001
        self.centroidWidth = 20
        self.numCohFrames = 16
        self.numIncohFrames = 100
        self.frameClockPeriod = 0.010
        self.expTime = 0.004
        self.quadRows = 1
        self.quadCols = 5
        self.quadRowOffset = 34
        self.quadColOffset = 45
        self.specRowOffset = 0
        self.specColOffset = 0
        self.numReadsPerPixel = 1
        self.specWave = np.array([1550e-9, 1596e-9, 1644e-9, 1695e-9, 1750e-9])
        self.modStart = 0.0
        self.modPeriod = 0.2
        self.modAmp = 1960e-9
        self._Init()

    def _Init(self):
        # Initialize attributes
        self.decayExp = exp(-float(self.numCohFrames)
                             / float(self.numIncohFrames))
        self.decayConst = (1. - self.decayExp)
        self.trialGD = np.zeros((self.numGDTrials,))
        for p in range(self.numGDTrials):
            self.trialGD[p] = (float(p - self.numGDTrials/2)
                               * self.trialGDScale 
                               / (1/np.amin(self.specWave) 
                                  - 1./np.amax(self.specWave)))
        self._CalculateTiming()
        self.F1 = np.zeros((self.numChannels,), np.complex)
        self.modExpCentre = np.zeros((self.numCohFrames,))
        self.time = np.zeros((self.numCohFrames,))
        self.results = FTBaselineResults(self.numChannels, self.numGDTrials,
                                         self.numCohFrames)
        # Precompute gdBasis
        self.gdBasis = np.zeros((self.numGDTrials, self.numChannels),
                                np.complex)
        chromaticTerm = 1.0
        for p in range(self.numGDTrials):
            for j in range(self.numChannels):
                win = welch(j, self.numChannels)
                wave = self.specWave[j]
                if j < self.numChannels-1:
                    wave1 = self.specWave[j+1]
                else:
                    wave1 = (self.specWave[j]
                             + (self.specWave[j] - self.specWave[j-1]))
                deltaSigma = 1./wave - 1./wave1
                self.gdBasis[p,j] = (win * cos(-2*pi/wave*self.trialGD[p])
                                     * chromaticTerm * 1./deltaSigma)
                self.gdBasis[p,j] += (1.0j 
                                      * win * sin(-2*pi/wave*self.trialGD[p])
                                      * chromaticTerm * 1./deltaSigma)

    def _CalculateTiming(self):
        triggerDelay = 5.04e-6
        triggerResetDelay = 160e-9
        pixelProcessTime = 223e-9
        self.nextRowTime = 1160.0e-9
        self.colSkipTime = 520.0e-9
        self.pixelReadTime = 1240.0e-9
        cdsReadTime = ((self.quadRowOffset + self.quadRows - 1)
                       * self.nextRowTime)
        # Time to skip all the pixels above the region of interest
        cdsReadTime +=  self.quadColOffset * self.quadRows * self.colSkipTime
        # Time to read the pixels of interest
        cdsReadTime += (self.quadCols * self.quadRows
                        * self.numReadsPerPixel * self.pixelReadTime)
        skipTime = (self.quadRowOffset * self.nextRowTime 
                    + self.quadColOffset * self.colSkipTime)
        self.cdsReadStart = skipTime + triggerDelay + triggerResetDelay
        self.expReadStart = (self.cdsReadStart + cdsReadTime + self.expTime
                             + skipTime
                             + self.quadRows * self.quadCols * pixelProcessTime)

    def Process(self, data, time, startFrame):
        """Process entire fringe dataset.

        Parameters are:
        
        data -- numpy array of shape (numFrames, numChannels)

        time -- numpy array of frame timestamps

        startFrame -- index of first frame to process

        """
        self._Init()  # in case parameters changed since construction
        data = data[startFrame:]
        time = time[startFrame:]
        numFrames = data.shape[0]
        numCoh = numFrames/self.numCohFrames
        cohData = np.reshape(data[:numCoh*self.numCohFrames,:],
                             (numCoh, self.numCohFrames, -1))
        resultsLog = None
        self.results.avgGroupDelaySpectrum[:] = 0.0  # reset running average
        for i in range(numCoh):
            self._ComputeF1(cohData[i], i*self.numCohFrames)
            self._ComputeF2()
            self._ComputeF3()
            self._ComputeDelay()

            self.results.epoch = time[(i+1)*self.numCohFrames-1]
            if resultsLog is None:
                resultsLog = np.reshape(self.results.asarray(), (-1, 1))
            else:
                resultsLog = np.append(resultsLog, self.results.asarray())
            
        return resultsLog

    def _ComputeF1(self, data, frameCount):
        """Compute complex fringe amplitude F_1(j)."""
        self.F1[:] = 0.0j
        normF1 = np.zeros((self.numChannels,))
        sumFlux = np.zeros((self.numChannels,))
        modMin = 0.0
        modMax = 0.0
        data1 = data - np.mean(data,0)  # subtract mean over time axis
        for k in range(self.numCohFrames):

            # Calculate time to read region of interest up to central
            # spectral channel...
            # ...time to skip rows within region of interest
            tSpecRead = (self.specRowOffset * self.nextRowTime)
            # ...time to skip cols within region of interest
            tSpecRead += (self.specRowOffset * self.quadColOffset
                          * self.colSkipTime)
            # ...time to read the pixels of interest
            tSpecRead += ((self.specRowOffset*self.quadCols +
                           self.specColOffset) *
                          self.numReadsPerPixel * self.pixelReadTime)

            # Hence calculate exposure start and end times measured
            # from 1st frame clock of continuous sequence, for central
            # spectral channel (this implementation ignores the fact
            # that the readout time and hence modulator position is
            # different for each channel):
            tExpStart = ((frameCount+k) * self.frameClockPeriod
                         + self.cdsReadStart + tSpecRead)
            tExpEnd =   ((frameCount+k) * self.frameClockPeriod
                         + self.expReadStart + tSpecRead)

            # Calculate wavelength-independent (on assumptions above) terms
            win = welch(k, self.numCohFrames)
            tk = (tExpStart + tExpEnd)/2.0
            modExpCentre = triangle(self.modAmp, self.modPeriod,
                                    tk - self.modStart)
            modExpStart = triangle(self.modAmp, self.modPeriod,
                                   tExpStart - self.modStart)
            modExpEnd = triangle(self.modAmp, self.modPeriod,
                                 tExpEnd - self.modStart)
            deltaTime = (tExpEnd - tExpStart)
            deltaMod = (modExpEnd - modExpStart)
            if modExpCentre > modMax:
                modMax = modExpCentre
            if modExpCentre < modMin:
                modMin = modExpCentre

            # Save modulation for latest coherent integration
            self.modExpCentre[k] = modExpCentre
            self.time[k] = tk

            # Compute contribution to F1 from frame
            for j in range(self.numChannels):
                wave = self.specWave[j]
                self.F1[j] += (win * cos(-2.*pi/wave*modExpCentre)
                               * deltaTime/deltaMod * data1[k,j])
                self.F1[j] += (1.0j 
                               * win * sin(-2.*pi/wave*modExpCentre)
                               * deltaTime/deltaMod * data1[k,j])
                normF1[j] += win * fabs(deltaTime/deltaMod) * data[k,j]
                sumFlux[j] += data[k,j]

        # Compute flux and complex visibility results
        interval = (self.numCohFrames*self.frameClockPeriod)
        for j in range(self.numChannels):
            self.results.flux[j] = sumFlux[j]/interval
            self.results.visibility[j] = 2.0*np.absolute(self.F1[j])/normF1[j]
            self.results.phase[j] = atan2(self.F1[j].imag, self.F1[j].real)

    def _ComputeF2(self):
        """Compute instantaneous group delay spectrum F_2(p)."""
        self.results.trialGD = self.trialGD
        self.results.groupDelaySpectrum[:] = 0.0j
        for p in range(self.numGDTrials):
            for j in range(self.numChannels):
                self.results.groupDelaySpectrum[p] += (self.gdBasis[p,j]
                                                       * self.F1[j])

    def _ComputeF3(self):
        """Compute running average group delay spectrum F_3."""
        self.results.avgGroupDelaySpectrum *= self.decayExp
        self.results.avgGroupDelaySpectrum += (self.decayConst * (np.absolute(
                    self.results.groupDelaySpectrum)**2.))

    def _ComputeDelay(self):
        """Compute delay estimate from centroid of average GD spectrum F_3."""
        maxPowIndex = np.argmax(self.results.avgGroupDelaySpectrum)
        if (maxPowIndex > self.centroidWidth/2
            and maxPowIndex < self.numGDTrials - self.centroidWidth/2):
            tot = 0.
            centroid = 0.
            for offset in range(-self.centroidWidth/2, self.centroidWidth/2):
                val = self.results.avgGroupDelaySpectrum[maxPowIndex+offset]
                tot += val
                centroid += (maxPowIndex + offset) * val
            centroid /= tot
            gdCentroid = (((1. - centroid + floor(centroid)) *
                           self.trialGD[int(floor(centroid))]) + 
                          ((centroid - floor(centroid)) * 
                           self.trialGD[int(floor(centroid)) + 1]))
            self.results.estDelay = gdCentroid
        else:
            self.results.estDelay = self.trialGD[maxPowIndex]


import matplotlib.pyplot as plt
import pylab

def _test():
    # Read fringe data and corresponding fringe engine logfile
    bg = get_background('bg.fits')
    # don't subtract background, for consistency with fringe engine
    bg[:,:] = 0.
    data, time = get_data('step01.fits', 1, 5, 1, 4, bg)
    log0 = new_log_data('step01.log')
    # empirical time offset to match results
    log0start = log0['epoch'][0] + 7.2

    # Process fringe data
    engine = BasdenEngine()
    # override default parameters
    engine.trialGDScale = 0.001
    engine.numCohFrames = 10
    engine.numIncohFrames = 10
    engine.quadRows = 3
    engine.quadCols = 10
    engine.quadRowOffset = 67
    engine.quadColOffset = 100
    engine.specRowOffset = 1
    engine.specColOffset = 4
    engine.specWave = np.array([1801.8e-9, 1769.9e-9,
                                1739.1e-9, 1680.7e-9, 1652.9e-9])
    engine.modPeriod = 0.2
    engine.modAmp = 1960.0e-9
    # begin processing at guessed start of modulation period
    log = engine.Process(data, time, 15)

    # Plot results and logged results from fringe engine
    log0epoch = log0['epoch'] - log0start
    plt.figure()
    plt.plot(time, data[:, 0], '-r+')
    plt.plot(time, data[:, 4], '--bo')
    plt.title('Data')
    plt.figure()
    plt.plot(log0epoch, log0['flux_1'], 'r')
    plt.plot(log0epoch, log0['flux_3'], 'g')
    plt.plot(log0epoch, log0['flux_5'], 'b')
    plt.plot(log['epoch'] - log['epoch'][0], log['flux_3'], 'k')
    plt.title('Flux_3')
    plt.figure()
    plt.plot(log0epoch, log0['visibility_1'], 'r')
    plt.plot(log0epoch, log0['visibility_3'], 'g')
    plt.plot(log0epoch, log0['visibility_5'], 'b')
    plt.plot(log['epoch'] - log['epoch'][0], log['visibility_3'], 'k')
    plt.title('Visibility_3')
    plt.figure()
    plt.plot(log0epoch[::20], log0['phase_1'][::20], 'r')
    plt.plot(log0epoch[::20], log0['phase_3'][::20], 'g')
    plt.plot(log0epoch[::20], log0['phase_5'][::20], 'b')
    plt.plot(log['epoch'][::2] - log['epoch'][0], log['phase_3'][::2], 'k')
    plt.title('Phase_3')
    plt.figure()
    plt.plot(log0epoch[10::20], log0['phase_1'][10::20], 'r')
    plt.plot(log0epoch[10::20], log0['phase_3'][10::20], 'g')
    plt.plot(log0epoch[10::20], log0['phase_5'][10::20], 'b')
    plt.plot(log['epoch'][1::2] - log['epoch'][0], log['phase_3'][1::2], 'k')
    plt.title('Phase_3')
    plt.figure()
    plt.plot(log0epoch, log0['estDelay'], 'g')
    plt.plot(log['epoch'] - log['epoch'][0], log['estDelay'], 'k')
    plt.title('estDelay')

    pylab.show()

if __name__ == '__main__':
    _test()
