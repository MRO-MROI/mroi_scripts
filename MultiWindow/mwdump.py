import numpy as np
import pylab
import matplotlib.pyplot as plt

def mwData(fname):
    fid = open(fname, 'r')

    data = fid.readlines()
    #nframes is -1 as the last recorded frame has no data
    nframes = (len(data))-1
    print nframes, " frames"
    
    frames = np.zeros((nframes,2))
    data1  = np.zeros((nframes,5))
    data2  = np.zeros((nframes,5))
    data3  = np.zeros((nframes,5))
    data4  = np.zeros((nframes,5))
    data5  = np.zeros((nframes,5))
       
   # print frames range is -4 as the last 4 lines of the file has no data
    for i in xrange(len(data)-4):
        if i%4 == 1:
            #get rid of extra characters
            frame = data[i].strip('\n').split('\t')
            #extract frame number, iterator is based on frame number as 
            #well, otherwise only every 4th frame is filled in frames
            frames[int(frame[0])-1][0] = int(frame[0])
            #extract epoch
            frames[int(frame[0])-1][1] = float(frame[1])
            
        elif i%4 == 3:
            lines = data[i].strip('\n').rstrip().split(' ')
            for l in xrange(len(lines)):
				 lines[l] = int(lines[l], 16)
            lines = np.array(lines)
            #L1-L5 include all 4 quadrants for any given line
            #L1 = lines[0:20].reshape((5,4))
            #L2 = lines[20:40].reshape((5,4))
            #L3 = lines[40:60].reshape((5,4))
            #L4 = lines[60:80].reshape((5,4))
            #L5 = lines[80:100].reshape((5,4))
            
            data1[int(frame[0])-1][:] = L1
            data2[int(frame[0])-1][:] = L2
            data3[int(frame[0])-1][:] = L3
            data4[int(frame[0])-1][:] = L4
            data5[int(frame[0])-1][:] = L5
        
    return [data1, data2, data3, data4, data5]
    
def plotmw(d):
    #d is data
    frames = xrange(len(d[0]))
    
    fig = plt.figure()
    
    ax1 = plt.subplot(511)
    ax2 = plt.subplot(512)
    ax3 = plt.subplot(513)
    ax4 = plt.subplot(514)
    ax5 = plt.subplot(515)

    ax1.plot(frames, d[0][:,q-1])
    ax1.set_ylim((-4000, 40000))
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Count')
    ax1.set_title('Line 1')
    ax1.legend(["Pixel 1", "Pixel 2", "Pixel 3", "Pixel 4", "Pixel 5"])

    ax2.plot(frames, d[1][:,q-1])
    ax2.set_ylim((-4000, 40000))
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Count')
    ax2.set_title('Line 2')
    ax2.legend(["Pixel 1", "Pixel 2", "Pixel 3", "Pixel 4", "Pixel 5"])
    
    ax3.plot(frames, d[2][:,q-1])
    ax3.set_ylim((-4000, 40000))
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Count')
    ax3.set_title('Line 3')
    ax3.legend(["Pixel 1", "Pixel 2", "Pixel 3", "Pixel 4", "Pixel 5"])
    
    ax4.plot(frames, d[3][:,q-1])
    ax4.set_ylim((-4000, 40000))
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Count')
    ax4.set_title('Line 4')
    ax4.legend(["Pixel 1", "Pixel 2", "Pixel 3", "Pixel 4", "Pixel 5"])
    
    ax5.plot(frames, d[4][:,q-1])
    ax5.set_ylim((-4000, 40000))
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('Count')
    ax5.set_title('Line 5')
    ax5.legend(["Pixel 1", "Pixel 2", "Pixel 3", "Pixel 4", "Pixel 5"])
    
    fig.subplots_adjust(hspace = 0.6)
    
if __name__ == "__main__":
    mw = mwData()	
    plotmw(mw)
    	
