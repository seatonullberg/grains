import os
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt

from grains import errors


class GrainsAnalyzer(object):
    """Calculates statistics about grains in a micrograph.

    Detemines the number and area distribution of grains in a given image.
    Text file summarization is built-in, but optional histogram and 
    marked-up image exports are also supported.
    
    Attributes:
        input_fn:       filename of the image being analyzed.
        height_microns: height of the image content in micrometers.
        width_microns:  width of the image content in micrometers.

        summary_fn:     filename to write the text summary to.
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

        extensionless = "".join(self.input_fn.split(".")[:-1])
        self.summary_fn = "{}.summary.txt".format(extensionless)
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

    def write_summary(self):
        """Writes the summary text file"""

        with open(self.summary_fn, "w") as f:
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
