# Blake Lawyer - graphvid.py
# Creates a graph of a video using superpixels. Inspired by Eitan Kosman and Dotan Di Castro GraphVid paper.
# https://arxiv.org/abs/2207.01375

import math
import networkx as nx
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
import numpy as np
import cv2
from skimage.measure import regionprops
import matplotlib.pyplot as plt

# Basic distance function for computing distance between 2 superpixel centroids.
def distance(x1, y1, x2, y2):
	x_diff = x2 - x1
	y_diff = y2 - y1
	x_sqr = x_diff ** 2
	y_sqr = y_diff ** 2
	sum_sqr = x_sqr + y_sqr
	sqr = math.sqrt(sum_sqr)
	return math.floor(sqr)

# Returns a "colorfulness" value used for node labels, and the average RGB of a superpixel.
def segment_colorfulness(image, mask):
	img_mask = image[np.where(mask == 0)]
	img_avg = np.mean(img_mask, axis=(0, 1))
	c = image.mean(axis=(0, 1))
	(B, G, R) = cv2.split(image.astype("float"))
	R = np.ma.masked_array(R, mask=mask)
	G = np.ma.masked_array(G, mask=mask)
	B = np.ma.masked_array(B, mask=mask)
	rg = np.absolute(R - G)
	yb = np.absolute(0.5 * (R + G) - B)
	stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
	meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
	return (stdRoot + (0.3 * meanRoot)), math.floor(img_avg)

# Create the capture object. This is while your source video goes.
capture = cv2.VideoCapture("video")
# Initialize a NetworkX graph.
G = nx.Graph()
# Initialize variables for past superpixels, the frame each superpixel is from, and the color of that superpixel.
past_superpixels = []
n_frame = []
n_color = []
f = 1
# While there are readable frames in the video.
while True:

	# Clear the current superpixel list and grab the next frame.
	current_superpixels = []
	ret, frame = capture.read()

	# If the frame was read correctly.
	if ret:
		# Resize the frame to fit on the screen when/if displayed. (display code at bottom)
		frame = cv2.resize(frame, (640, 360))
		orig = frame
		vis = np.zeros(orig.shape[:2], dtype="float")
		image = frame

		# Use the SLIC algorithm to segment the image into n many superpixels.
		segments = slic(img_as_float(image), n_segments=10,
			slic_zero=True)

		# For each unique superpixel..
		for v in np.unique(segments):
			# Create the mask corresponding to it.
			mask = np.ones(image.shape[:2])
			mask[segments == v] = 0
			# Get the "colorfulness" value and color.
			C, color = segment_colorfulness(orig, mask)
			# Keep track of all current frame's superpixels for graph additions and connections.
			current_superpixels.append((math.floor(C), color))
			vis[segments == v] = C

		vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")
		alpha = 0.6
		overlay = np.dstack([vis] * 3)
		output = orig.copy()
		cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

		# List of superpixels.
		regions = regionprops(segments)

		# Add unique superpixels as nodes in the graph and track their frame and color.
		# Node size is determined by the frame number * constant to show temporality.
		for n in current_superpixels:
			if not G.has_node(n[1]):
				n_frame.append(f*25)
				n_color.append(n[1])
			G.add_node(n[1])

		# Connect all superpixels that within a distance threshold of each other.
		# This is a crude way to determine the neighboring superpixels. Proper algorithms exist.
		for sp_region, superpixel in enumerate(current_superpixels):
			for p_region, pixel in enumerate(current_superpixels):
				x1, y1 = regions[sp_region].centroid
				x2, y2 = regions[p_region].centroid
				if distance(x1, y1, x2, y2) < 200:
					if superpixel[1] != pixel[1]:
						G.add_edge(superpixel[1], pixel[1])

		# Temporal connections between superpixels in sequential frames.
		if past_superpixels:
			for psp in past_superpixels:
				for csp in current_superpixels:
					if csp[1] != psp[1]:
						G.add_edge(psp[1], csp[1])

		# Current superpixels become the previous superpixels during the next iteration.
		past_superpixels = current_superpixels.copy()
		f += 1
	else:
		# If there are no more frames to read, break out of the loop.
		break

# Draw and show the network with specified node size, labels, and node color.
nx.draw_networkx(G, node_size=n_frame, with_labels=True, node_color=n_color)
plt.show()


"""
CODE TO DISPLAY THE SUPERPIXEL SEGMENTATION
#cv2.imshow("Input", orig)
#cv2.imshow("Visualization", vis)
#cv2.imshow("Output", output)
#cv2.waitKey(0)
"""