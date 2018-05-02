##	Alex Leong adl88
##	ASTRO 3334 Spring 2018
##	Lab 7: Hertzsprung-Russell Diagram

import sys
import pickle as p
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from numpy import log10,sqrt
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from photutils import CircularAperture, DAOStarFinder, datasets


starDataBases = ["","starData1.pkl", "starData2.pkl","starData3.pkl"]
neighborDataBases = ["","closestNeighbors1.pkl", "closestNeighbors2.pkl","closestNeighbors3.pkl"]


def analyzeImages(X):
	# create database
	f = open(starDataBases[X], "wb")

	# retrieve data
	hdu = fits.open('./HRData.fits')
	data814 = hdu[0].data[0,:,:]
	data275 = hdu[0].data[2,:,:]
	hdu.close()


	mean814, median814, std814 = sigma_clipped_stats(data814, sigma=3.0, iters=5)
	daofind814 = DAOStarFinder(fwhm=3.0, threshold=4.*std814)

	mean275, median275, std275 = sigma_clipped_stats(data275, sigma=3.0, iters=5)
	daofind275 = DAOStarFinder(fwhm=3.0, threshold=4.*std275)

	
	## perform photometry analysis
	# both apertures
	if (X == 1):
		print("Find stars from F814W Filter Data")
		sources814 = daofind814(data814 - median814)
		print("Finished F814W Search")

		print("Find stars from F275W Filter Data")
		sources275 = daofind275(data275 - median275)
		print("Finished F275 Search")

		positions814 = (sources814['xcentroid'], sources814['ycentroid'])
		apertures814 = CircularAperture(positions814, r=4.)
		photom814 = apertures814.do_photometry(data814 - median814)

		positions275 = (sources275['xcentroid'], sources275['ycentroid'])
		apertures275 = CircularAperture(positions275, r=4.)
		photom275 = apertures275.do_photometry(data275 - median275)

	# apertures from infrared
	elif (X == 2):
		print("Find stars from F814W Filter Data")
		sources814 = daofind814(data814 - median814)
		print("Finished F814W Search")

		positions814 = (sources814['xcentroid'], sources814['ycentroid'])
		apertures814 = CircularAperture(positions814, r=4.)
		photom814 = apertures814.do_photometry(data814 - median814)

		positions275 = positions814
		photom275 = apertures814.do_photometry(data275 - median275)

	# apertures from ultraviolet
	elif (X == 3):
		print("Find stars from F275W Filter Data")
		sources275 = daofind275(data275 - median275)
		print("Finished F275 Search")

		positions275 = (sources275['xcentroid'], sources275['ycentroid'])
		apertures275 = CircularAperture(positions275, r=4.)
		photom275 = apertures275.do_photometry(data275 - median275)

		positions814 = positions275
		photom814 = apertures275.do_photometry(data814 - median814)

	else:
		print("invalid database selection")
		return

	# store data
	data = (positions814, photom814, positions275, photom275)
	p.dump(data, f)
	f.close()

	return data


def getNeighbors(X,positions814,positions275):
	f = open(neighborDataBases[X], "wb")
	positions814 = [list(elem) for elem in zip(positions814[0], positions814[1])]
	positions275 = [list(elem) for elem in zip(positions275[0], positions275[1])]
	searchTree = KDTree(positions814)
	result = searchTree.query(positions275)
	p.dump(result, f)
	f.close()

	return result


def main(X):

	# Load photometry data if database exists, analyze and store FITS data in database otherwise
	try:
		f = open(starDataBases[X], "rb")
	except IOError:
		f = open(starDataBases[X], "wb")
	try:
		(positions814, photom814, positions275, photom275) = p.load(f)
	except EOFError:
		(positions814, photom814, positions275, photom275) = analyzeImages()
	except IOError:
		(positions814, photom814, positions275, photom275) = analyzeImages()
	f.close()

	# Load neighbor data if database exists, find neighbors and store data in database otherwise
	try:
		f = open(neighborDataBases[X], "rb")
	except IOError:
		f = open(neighborDataBases[X], "wb")
	try:
		neighbors = p.load(f)
	except EOFError:
		neighbors = getNeighbors(positions814,positions275)
	except IOError:
		neighbors = getNeighbors(positions814,positions275)
	f.close()


	# upper bound on pixel distance between two aperture centers to be
	# considered containing the same star (no effect if using database 2 or 3)
	eps = 1	
	
	graphAlpha = 0.4

	PHOTZPT = -2.1100000E+01
	PHOTFLAM = 1.4772700E-19

	numStars = len(neighbors[0])
	print("Total number of stars identified: " + str(numStars))

	xMagLower = 0
	xMagUpper = 10

	yMagLower = 13
	yMagUpper = 23
	
	# Convert data for plotting
	xPts = []
	yPts = []
	numValidResults = 0
	for i in range(numStars):
		if (neighbors[0][i] < eps):
			linFlux814 = photom814[0][neighbors[1][i]]
			linFlux275 = photom275[0][i]

			if (linFlux814 > 0 and linFlux275 > 0):

				# Linear flux to calibrated instrumental magnitude
				mag814 = (-2.5 * log10(linFlux814)) - (2.5 * log10(PHOTFLAM)) + PHOTZPT
				mag275 = (-2.5 * log10(linFlux275)) - (2.5 * log10(PHOTFLAM)) + PHOTZPT

				xPt = mag275 - mag814
				yPt = mag814
				
				if (xPt > xMagLower and xPt < xMagUpper and yPt > yMagLower and yPt < yMagUpper):
					xPts.append(xPt)
					yPts.append(yPt)
					numValidResults += 1
				
	print("Number of valid results: " + str(numValidResults))


	# Create H-R Diagram from Data
	s = [((yMagUpper - n)/(yMagUpper - yMagLower + 1))**2 for n in yPts] # point sizes
	plt.scatter(xPts,yPts,marker="o",alpha=graphAlpha,s=s,c=xPts,cmap=plt.cm.coolwarm)
	plt.xlabel("[F275W] - [F814W]")
	plt.ylabel("[F814W]")
	if (X == 1): plt.title("Hertzsprung-Russell Diagram (Point Matched)")
	elif (X == 2): plt.title("Hertzsprung-Russell Diagram (Identical Apertures from F814W)")
	elif (X == 3): plt.title("Hertzsprung-Russell Diagram (Identical Apertures from F275W)")
	plt.ylim((yMagLower,yMagUpper))
	plt.xlim((xMagLower,xMagUpper))
	ax = plt.gca()
	ax.invert_yaxis()
	ax.set_facecolor((0,0,0))
	
	plt.show()


if __name__ == "__main__":
	# 1 : different apertures, point matched
	# 2 : same apertures (infrared)
	# 3 : same apertures (ultraviolet))
	try:
		if (len(sys.argv) != 2 or sys.argv[1] not in ["1", "2", "3"]): raise ValueError
		main(int(sys.argv[1]))
	except ValueError:
		print("Invalid Database Commandline Argument (1, 2 or 3)")
