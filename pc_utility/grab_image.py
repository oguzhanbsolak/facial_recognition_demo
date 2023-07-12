import serial
import time
from datetime import datetime
import os
import sys
import string
import comManager
import imgConverter
import cv2
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) == 3:
	comport  = sys.argv[1]
	baudRate = sys.argv[2]
else:
	comport  = sys.argv[1]
	# Setup the default baudrate.
	baudRate = 115200  # 115200

READ_TIMEOUT = 15

def ByteToHex( ch ):
	try:
		ch = ord(ch)
	except:
		ch = 0
	return ch

def print_sep_line( ch ):
	line = ch * 80
	print(line, flush=True)

#----------------------------------------------

print("Image Reader Started")
print_sep_line('-')
r = [] 
g = [] 
b = []
retVal = comManager.init(comport, baudRate)
if retVal != 0:
	print ("comport open failed. Please check %s status\n" % comport)
	sys.exit()
while True:
	if( comManager.find_sync() == 1 ):
		i = 0

		print ("\n\n***Sync word found***", flush=True)
		print ("Reading image bytes, please wait...", flush=True)
		while(i < 160*120):
	

			r.append(comManager.read(1))
			g.append(comManager.read(1))
			b.append(comManager.read(1))
		
			i = i + 1
			if i % 1000 == 0:
				print(i)
	if (len(r) != 0):	
		print(len(r), len(g), len(b))
		img = np.zeros((160, 120, 3), np.uint8)
		for y in range(160):
			for x in range(120):
				img[y][x][0] = int.from_bytes(r[y*120+x], "little")
				img[y][x][1] = int.from_bytes(g[y*120+x], "little")
				img[y][x][2] = int.from_bytes(b[y*120+x], "little")
		r = [] 
		g = [] 
		b = []
		
		print(img.min(), img.max())
		plt.imshow(img)
		plt.show()
