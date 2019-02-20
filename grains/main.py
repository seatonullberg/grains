import os
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt

from grains import errors


class GrainsAnalyzer(object):
    """Calculates statistics about grains in a micrograph.

    Detemines the number and area distribution of grains in a given image.
    Capable of writing out a summary in 3 formats:
        - markup of the original image with superimposed grain centroids
        - histogram of the area distribution
        - text file summarizing the analysis
    
    Attributes:
        input_fn:       filename of the image being analyzed.
        height_microns: height of the image content in micrometers.
        width_microns:  width of the image content in micrometers.

        img_out_fn:     filename to write the markup image to.
        plot_out_fn:    filename to write the histogram plot to.
        stats_out_fn:   filename to write the text summary to.
        
        base_image:     input image as an array.
        working_image:  input_image as an array after preprocessing.
        contours:       OpenCV contours detected in the working_image.
        moments:        OpenCV moments detected in the working_image.
        centroids:      OpenCV centroids detected in the working_image.
    """

    def __init__(self, input_fn, height_microns, width_microns):
        """Inits GrainsAnalyzer with required information"""
        
        self.input_fn = input_fn
        self.height_microns = height_microns
        self.width_microns = width_microns
        self.img_out_fn = "{}.grains.image.png".format(self.input_fn)
        self.plot_out_fn = "{}.grains.plot.png".format(self.input_fn)
        self.stats_out_fn = "{}.grains.stats.txt".format(self.input_fn)
        self.base_image = cv2.imread(self.input_fn)
        self.working_image = None
        self.contours = None
        self.moments = None
        self.centroids = None

    @property
    def total_count(self):
        """Returns number of grains detected
        
        Raises:
            UnsetContoursError
        """

        if self.contours is None:
            raise errors.UnsetContoursError("self.contours has not yet been set.\nRun Grains().set_contours() to do so.")
        return len(self.contours)

    @property
    def area_mean(self):
        """Returns the mean grain area in micrometers squared
        
        In accordance with ASTM E112 planimetric method.
        """

        mean = np.sum(self.contour_areas_microns())/len(self.contour_areas_microns())
        return mean

    @property
    def area_variance(self):
        """Returns the variance of grain area in micrometers squared"""

        var = np.var(self.contour_areas_microns())
        return var

    @property
    def area_stdev(self):
        """Returns the standard deviation of grain area in micrometers squared"""

        stdev = np.std(self.contour_areas_microns())
        return stdev

    def preprocess_image(self, thresh=None, maxval=None, thresh_type=None, kernel=None, iterations=None):
        """Prepares the image for contour detection by binarizing color and reducing noise

        Args:
            thresh:     thresh passed to cv2.threshold()
            maxval:     maxval passed to cv2.threshold()
            kernel:     kernel passed to cv2.threshold()
            iterations: iterations passed to cv2.morphologyEx() for the `Open` operation 
        """

        # process args
        if thresh is None:
            thresh = 50
        if maxval is None:
            maxval = 255
        if thresh_type is None:
            thresh_type = cv2.THRESH_BINARY
        if kernel is None:
            kernel = np.ones((3,3), np.uint8)
        if iterations is None:
            iterations = 6
        
        # do thresholding and noise reduction with the `MORPH_OPEN` operation
        img = cv2.cvtColor(self.base_image, cv2.COLOR_BGR2GRAY)
        _ret, img = cv2.threshold(img, thresh, maxval, thresh_type)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
        self.working_image = img

    def set_contours(self, mode=None, method=None):
        """Detects contours and sets the self.contours attribute

        Args:
            mode:   mode passed to cv2.findContours()
            method: method passed to cv2.findContours()
        
        Raises:
            UnsetWorkingImageError
        """

        # check runtime order
        if self.working_image is None:
            raise errors.UnsetWorkingImageError("self.working_image has not yet been set.\nRun Grains().preprocess_image() to do so.")
        
        # process args
        if mode is None:
            mode = cv2.RETR_TREE
        if method is None:
            method = cv2.CHAIN_APPROX_SIMPLE
        
        # locate contours throw away hierarchy
        contours, _ = cv2.findContours(self.working_image, mode, method)
        self.contours = contours

    def set_moments(self):
        """Extracts moments from OpenCV contours

        Raises:
            UnsetContoursError
        """

        # check runtime order
        if self.contours is None:
            raise errors.UnsetContoursError("self.contours has not yet been set.\nRun Grains().set_contours() to do so.")
        
        # collect all moments
        moments = []
        for c in self.contours:
            m = cv2.moments(c)
            moments.append(m)
        self.moments = moments

    def set_centroids(self):
        """Extracts centroids from OpenCV moments

        Raises:
            UnsetMomentsError
        """

        # check runtime order
        if self.moments is None:
            raise errors.UnsetMomentsError("self.moments has not yet been set.\nRun Grains().set_moments() to do so.")
        
        # collect all centroids in list of dicts
        centroids = []
        for m in self.moments:
            coords = {}
            if m["m00"] == 0:
                coords["x"] = 0
                coords["y"] = 0
            else:
                coords["x"] = int(m["m10"] / m["m00"])
                coords["y"] = int(m["m01"] / m["m00"])
            centroids.append(coords)
        self.centroids = centroids         

    def contour_areas_microns(self):
        """Calculates the area of detected contours in square micrometers
        
        Returns:
            list of floats
        
        Raises:
            UnsetContoursError
        """

        # check runtime order
        if self.contours is None:
            raise errors.UnsetContoursError("self.contours has not yet been set.\nRun Grains().set_contours to do so.")
        # collect all areas
        areas = []
        for c in self.contours:
            pix_area = cv2.contourArea(c)
            microns_area = self._pix_area_to_microns_area(pix_area)
            areas.append(microns_area)
        return areas

    def write_img_out(self):
        """Writes the marked-up image file
        
        Raises:
            UnsetCentroidsError
        """

        if self.centroids is None:
            raise errors.UnsetCentroidsError("self.centroids has not yet been set.\nRun Grains().set_centroids() to do so.")
        
        img = copy.deepcopy(self.base_image)
        for i, c in enumerate(self.centroids):
            cv2.circle(img, (c["x"], c["y"]), 3, (0,0,255), thickness=-3)
        cv2.imwrite(self.img_out_fn, img)

    def write_hist_out(self, bins=None):
        """Writes the histogram file
        
        Args:
            bins: bins passed to plt.hist()
        """

        # process args
        if bins is None:
            bins = 20

        # plot a histogram of the areas
        areas = self.contour_areas_microns()
        plt.hist(x=areas, bins=bins)
        plt.title("Grain Area Distribution")
        plt.xlabel("microns squared")
        plt.ylabel("counts")
        plt.tight_layout()
        plt.savefig(self.plot_out_fn)

    def write_stats_out(self):
        """Writes the summary text file"""

        with open(self.stats_out_fn, "w") as f:
            input_fn = os.path.basename(self.input_fn)
            f.write("Automatically Generated by grains\n\n")
            f.write("Input filename: {}\n".format(input_fn))
            f.write("Number of grains: {}\n".format(self.total_count))
            mean = round(self.area_mean, 2)
            f.write("Grain area mean: {} um^2\n".format(mean))
            var = round(self.area_variance, 2)
            f.write("Grain area variance: {} um^2\n".format(var))
            stdev = round(self.area_stdev, 2)
            f.write("Grain area standard deviation: {} um^2\n".format(stdev))

    def _pix_area_to_microns_area(self, target_area_pix):
        image_area_microns = self.height_microns * self.width_microns
        image_area_pix = self.working_image.size
        target_area_microns = (target_area_pix/image_area_pix) * image_area_microns
        return target_area_microns
