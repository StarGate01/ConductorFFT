#!/usr/bin/python3

import numpy as np
import cv2
import time
import collections
import numpy.fft as fft
from scipy.signal import blackmanharris, find_peaks


# Setup video source
cap = cv2.VideoCapture(0) #('/home/christoph/Desktop/Probevideo_Dirigierbeispiele.webm') # (0)
samp_rate = cap.get(cv2.CAP_PROP_FPS)
print("ConductorFFT is running, press Q to exit")
print("VideoCapture sample rate: " + str(samp_rate))

# Setup FFT
fft_len = 0
xf = []
fft_queue = None
def fft_change(x):
    global fft_len, xf, fft_queue
    fft_len = 2 ** x
    xf = fft.fftfreq(fft_len) * samp_rate
    fft_queue = collections.deque(maxlen=fft_len)
    for i in range(fft_len): fft_queue.append(0)
fft_change(8)

# Setup windows control elements
def nothing(x): pass
cv2.namedWindow('webcam', flags=cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('webcam', 800,600)
cv2.namedWindow('process', flags=cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('process', 800,600)
cv2.createTrackbar('H_min', 'process', 164, 255, nothing)
cv2.createTrackbar('S_min', 'process', 10, 255, nothing)
cv2.createTrackbar('V_min', 'process', 132, 255, nothing)
cv2.createTrackbar('H_max', 'process', 174, 255, nothing)
cv2.createTrackbar('S_max', 'process', 253, 255, nothing)
cv2.createTrackbar('V_max', 'process', 255, 255, nothing)
cv2.createTrackbar('FFT_exp', 'webcam', 8, 14, fft_change)
cv2.createTrackbar('FFT_tresh', 'webcam', 20, 200, nothing)

# Processing loop
while(True):
    # Read frame and frame info
    ret, frame = cap.read()
    height, width, channels = frame.shape
    if ret:
        # Image preprocessing - blur and convert to HSV
        blurred = cv2.blur(frame, (8, 8))
        inter = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # Color tresholding
        lower_color = np.array([cv2.getTrackbarPos('H_min','process'), \
            cv2.getTrackbarPos('S_min','process'), cv2.getTrackbarPos('V_min','process')], dtype=np.uint8)
        upper_color = np.array([cv2.getTrackbarPos('H_max','process'), \
            cv2.getTrackbarPos('S_max','process'), cv2.getTrackbarPos('V_max','process')], dtype=np.uint8)
        mask = cv2.inRange(inter, lower_color, upper_color)
        # Patch holes and glitches
        kernel = np.ones((8,8), np.uint8)
        dilation =  cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Find center of largest blob
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilation, 4, cv2.CV_32S)
        largest_area = 0
        x = -1
        y = -1
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if(area > largest_area):
                largest_area = area
                x = centroids[i, 0]
                y = centroids[i, 1]

        # Feed Y position of blob into FFT buffer
        fft_queue.append(y)
        # Draw center of blob
        cv2.circle(frame, (int(x), int(y)), 10, (255, 255, 0), 3) 

        # Compute FFT spectrum using blackman-harris filter
        # Tuncate to N/2 (nyquist)
        yf = np.abs(fft.fft(fft_queue * blackmanharris(fft_len, False))[:fft_len//2])
        # Find FFT peaks and their frequencies
        peaks, _  = find_peaks(4 * yf / height, height=cv2.getTrackbarPos('FFT_tresh','webcam'))
        freqs = xf[peaks]

        # This function plots a line graph
        def plt(values, color):
            xpos = 0.0
            xpos_to = 0.0
            for i in range(1, len(values)):
                xpos_to += width / len(values)
                cv2.line(frame, (int(xpos),int(values[i-1])), \
                    (int(xpos_to),int(values[i])),color,2)
                xpos = xpos_to
        # Plot FFT buffer
        plt(fft_queue, (255, 0, 0))
        # Plot FFT tresholding line
        fft_tresh_l = height - (int((cv2.getTrackbarPos('FFT_tresh','webcam') * height / 4) / 50))
        cv2.line(frame, (0, fft_tresh_l), (width, fft_tresh_l), (0, 255, 0), 2)
        # Plot FFT spectrum
        plt(height - (yf / 50), (0, 255, 0))
        # Plot into text
        cv2.putText(frame, 'FFT max: ' + str(np.round(freqs, 2)), \
            (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Show image buffers
        cv2.imshow('webcam', frame)
        cv2.imshow('process', dilation)
        
        # Exit on 'Q' key
        if (cv2.waitKey(1) & 0xFF) == ord('q'): break
        time.sleep(1.0 / samp_rate)
    else:
        # Rewind video source
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Memory cleanup
cap.release()
cv2.destroyAllWindows()