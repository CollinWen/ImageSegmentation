'''
Collin Wen
4/22/19

Image Segmentation using K-means Clustering
'''

import cv2
import random
import numpy as np
import math


# k-means clustering on image given:
# 	k (# of centroids)
#	img (3d array (x,y,rgb))
# 	epoch (# number of training iterations)
#	fileName (output location of images)
#
# The k-means algorithm first defines centroids (random color pixel values),
# then determines which color values in the image are closest to each centroid.
# Once the closest centroid is found, that pixel will instead display the color
# of the centroid. Thus, the image is reduced to k colors. Then, centroids are
# readjusted to the average of all the pixels that display its color.
# This process is repeated for each epoch. Images from each epoch will be saved.
def kmeans(k, img, epoch, fileName):
	# save heigth and width of image
	height = img.shape[0]
	width = img.shape[1]

	# define centroids (random color values from the image)
	centroids = []

	randVals = random.sample(range(width*height), k)
	print(randVals)

	for i in randVals:
		pixel = img[i//width][i%width]
		dup = []
		for j in pixel:
			dup.append(j)

		centroids.append(dup)


	# define centroids (completely random color values)
	# for i in range(0,k):
	#	 centroids.append([random.randint(0,256), random.randint(0,256), random.randint(0,256)])

	new_img = np.zeros(shape=(height, width, 3))

	# training iteration
	for a in range(0, epoch):

		print('Epoch: ' + str(a))
		print('Centroids: ' + str(centroids))

		cluster_count = np.zeros(k)
		cluster_total = np.zeros(shape=(k,3))

		# iterates each row
		for i in range(0, height):

			# iterates each column
			for j in range(0, width):

				minDist = 2147483647.0
				centroid = -1

				# finds closest centroid
				for x in range(0, k):
					if distance(centroids[x], img[i][j]) < minDist:
						minDist = distance(centroids[x], img[i][j])
						centroid = x

				new_img[i][j] = centroids[centroid]

				# adds to cluster_total and cluster_count to find mean
				cluster_count[centroid] += 1
				for x in range(0, 3):
					cluster_total[centroid][x] += img[i][j][x]

		for i in range(0,k):
			for j in range(0,3):
				centroids[i][j] = round(cluster_total[i][j]/float(cluster_count[i]))

		cv2.imwrite('epoch' + str(a) + '.jpg', new_img)

	return new_img

# Returns the 3-dimensional distance between two rgb values
# 	p0 (first rgb array)
# 	p1 (second rbg array)
#
# The distance between two 3-dimensional points can be found by simply applying the distance formula.
def distance(p0, p1):
    return math.sqrt(float((p0[0] - p1[0])**2) + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)




# take input filename and output filename
inputFile = input('Enter input file: ')
outputFile = input('Enter output file: ')

# read image as numpy array
img = cv2.imread(inputFile)

# show image in separate window
# cv2.imshow('image',img)
# cv2.waitKey(0)

# perform kmeans algorithms on image and save new image to output file
new_img = kmeans(2,img,1, inputFile)
cv2.imwrite(outputFile, new_img)
