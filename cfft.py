#!/usr/bin/python3

import numpy as np
import cv2
import time
import collections
import numpy.fft as fft
import math
from scipy.signal import blackmanharris, find_peaks
import configparser
import sys, os
import argparse


# Setup config
config = configparser.ConfigParser()

# Store config to file on slider change
def config_change_fac(ns, name):
    global config
    def config_change(x):
        config[ns][name] = str(x)
        with open(configfile, 'w') as fhandle:
            config.write(fhandle)
    return config_change

# Setup video source
cap = cv2.VideoCapture(0) #('/home/christoph/Desktop/Probevideo_Dirigierbeispiele.webm') # (0)
samp_rate = cap.get(cv2.CAP_PROP_FPS)

# Setup FFT
fft_len = 0
xf = []
fft_queue = None
# Recalculate FFT parameters if needed
def fft_change(x):
    global fft_len, xf, fft_queue
    fft_len = 2 ** x
    xf = fft.fftfreq(fft_len) * samp_rate
    fft_queue = collections.deque(maxlen=fft_len)
    for i in range(fft_len): fft_queue.append(0)
    config_change_fac("webcam", "fft_exp")(x)

# Main program
def main(configfile):
    global fft_len, xf, fft_queue, config, cap, samp_rate

    print("ConductorFFT is running, press Q to exit")
    print("VideoCapture sample rate: " + str(samp_rate))

    # Setup windows
    cv2.namedWindow('webcam', flags=cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow('webcam', 800,600)
    cv2.namedWindow('process', flags=cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow('process', 800,600)
    cv2.namedWindow('gauge', flags=cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow('gauge', 400,400)
    # Setup sliders
    for entry in config["process"]:
        cv2.createTrackbar(entry, 'process', int(config["process"][entry]), 255, config_change_fac("process", entry))
    cv2.createTrackbar('fft_exp', 'webcam', int(config["webcam"]["fft_exp"]), 14, fft_change)
    cv2.createTrackbar('fft_tresh', 'webcam', int(config["webcam"]["fft_tresh"]), 200, config_change_fac("webcam", "fft_tresh"))
    for entry in config["gauge"]:
        cv2.createTrackbar(entry, 'gauge', int(config["gauge"][entry]), 300, config_change_fac("gauge", entry))
    # Init FFT and gauge image
    fft_change(int(config["webcam"]["fft_exp"]))
    gauge = np.zeros(shape=[400, 400, 3], dtype=np.uint8)

    # Processing loop
    while(True):
        # Read frame and frame info
        ret, frame = cap.read()
        height, width, _ = frame.shape
        if ret:
            # Time start 
            t_start = time.time()
            # Image preprocessing - blur and convert to HSV
            blurred = cv2.blur(frame, (8, 8))
            inter = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            # Color tresholding
            lower_color = np.array([cv2.getTrackbarPos('h_min','process'), \
                cv2.getTrackbarPos('s_min','process'), cv2.getTrackbarPos('v_min','process')], dtype=np.uint8)
            upper_color = np.array([cv2.getTrackbarPos('h_max','process'), \
                cv2.getTrackbarPos('s_max','process'), cv2.getTrackbarPos('v_max','process')], dtype=np.uint8)
            mask = cv2.inRange(inter, lower_color, upper_color)
            # Patch holes and glitches
            kernel = np.ones((8,8), np.uint8)
            dilation =  cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # Find center of largest blob
            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(dilation, 4, cv2.CV_32S)
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
            if(y <= 0):
                fft_queue.append(fft_queue[-1])
            else:
                fft_queue.append(y)

            # Compute FFT spectrum using blackman-harris filter
            # Tuncate to N/2 (nyquist)
            yf = np.abs(fft.fft(fft_queue * blackmanharris(fft_len, False))[:fft_len//2])
            # Find FFT peaks and their frequencies
            peaks, _  = find_peaks(4 * yf / height, height=cv2.getTrackbarPos('fft_tresh','webcam'))
            freqs = xf[peaks] * 60.0

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
            fft_tresh_l = height - (int((cv2.getTrackbarPos('fft_tresh','webcam') * height / 4) / 50))
            cv2.line(frame, (0, fft_tresh_l), (width, fft_tresh_l), (0, 255, 0), 2)
            # Plot FFT spectrum
            plt(height - (yf / 50), (0, 255, 0))
            # Draw center of blob
            cv2.circle(frame, (int(x), int(y)), 10, (255, 255, 0), 3) 
            # Plot info text
            resol = (samp_rate / fft_len) * 30.0
            cv2.putText(frame,  'BPM (+-' + str(np.round(resol, 2)) + '): ' + str(np.round(freqs, 2)), \
                (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Plot gauge
            gauge[:] = (0, 0, 0)
            gmin = cv2.getTrackbarPos('bpm_min','gauge')
            gmax = cv2.getTrackbarPos('bpm_max','gauge')
            # Plot gauge background and min, mid, max markers
            cv2.ellipse(gauge, (200, 200), (150, 150), 135, 0, 270, (255, 255, 0), 2)
            cv2.putText(gauge, str(gmin), (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(gauge, str(gmax), (305, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(gauge, str(np.round(gmin + ((gmax - gmin) / 2.0), 2)), \
                (165, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            if(len(freqs) > 0 and gmax > gmin):
                # Compute needle angle
                graw = freqs[0]
                gval = min(gmax, max(gmin, graw))
                gangle = (((gval - gmin) / (gmax - gmin)) * 270.0) + 135.0
                gx = (math.cos(math.pi * gangle / 180.0) * 150.0) + 200.0
                gy = (math.sin(math.pi * gangle / 180.0) * 150.0) + 200.0
                # Plot needle
                cv2.line(gauge, (200, 200), (int(gx), int(gy)), (0, 0, 255), 2)
                cv2.putText(gauge, str(np.round(graw, 2)), \
                    (165, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(gauge, (200, 200), 10, (255, 255, 0), -1)
            
            # Show image buffers
            cv2.imshow('webcam', frame)
            cv2.imshow('process', dilation)
            cv2.imshow('gauge', gauge)
            
            # Exit on 'Q' key
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break

            # Time end, compute delta time for constant sample rate
            t_end = time.time()
            elapsed = t_end - t_start
            delta = (1.0 / samp_rate) - elapsed
            if(delta < 0.0): 
                print("WARNING: Calculation to slow for framerate")
            else:
                time.sleep(delta)
        else:
            print("WARNING: Cannot aquire frame")

    # Memory cleanup
    cap.release()
    cv2.destroyAllWindows()


# Entrypoint
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description = "ConductorFFT")
    parser.add_argument("-c", "--configfile", help = "Specify configuration file", required = False, default = "config.ini")
    argument = parser.parse_args()

    # Ensure a configuration exists
    configfile = argument.configfile
    print("Config file: " + configfile)
    if(not os.path.exists(configfile)): 
        # Write default config
        config['process'] = {
            'h_min': 100,
            's_min': 100,
            'v_min': 100,
            'h_max': 200,
            's_max': 200,
            'v_max': 200
        }
        config['webcam'] = {
            'fft_exp': 8,
            'fft_tresh': 20
        }
        config['gauge'] = {
            'bpm_min': 30,
            'bpm_max': 160
        }
        # Write config to file
        with open(configfile, 'w') as fhandle:
            config.write(fhandle)
    else:
        # Read config from file
        config.read(configfile)

    # Start main program
    main(configfile)