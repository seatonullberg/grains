import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt


class Grains(object):

    def __init__(self, input_fn, height_microns, width_microns):
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
    def grain_count(self):
        return len(self.contours)

    @property
    def average_grain_area(self):
        mean = np.sum(self.contour_areas_microns())/len(self.contour_areas_microns())
        return mean

    @property
    def average_grain_area_ASTME112(self):
        raise NotImplementedError()

    def preprocess_image(self, thresh=None, maxval=None, thresh_type=None, kernel=None, iterations=None):
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
        # check runtime order
        if self.working_image is None:
            raise RuntimeError("self.working_image has not yet been set.\nRun Grains().preprocess_image() to do so.")
        
        # process args
        if mode is None:
            mode = cv2.RETR_TREE
        if method is None:
            method = cv2.CHAIN_APPROX_SIMPLE
        
        # locate contours throw away hierarchy
        contours, _ = cv2.findContours(self.working_image, mode, method)
        self.contours = contours

    def set_moments(self):
        # check runtime order
        if self.contours is None:
            raise RuntimeError("self.contours has not yet been set.\nRun Grains().set_contours() to do so.")
        
        # collect all moments
        moments = []
        for c in self.contours:
            m = cv2.moments(c)
            moments.append(m)
        self.moments = moments

    def set_centroids(self):
        # check runtime order
        if self.moments is None:
            raise RuntimeError("self.moments has not yet been set.\nRun Grains().set_moments() to do so.")
        
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
        # check runtime order
        if self.contours is None:
            raise RuntimeError("self.contours has not yet been set.\nRun Grains().set_contours to do so.")
        # collect all areas
        areas = []
        for c in self.contours:
            pix_area = cv2.contourArea(c)
            microns_area = self._pix_area_to_microns_area(pix_area)
            areas.append(microns_area)
        return areas

    def write_img_out(self):
        if self.centroids is None:
            raise RuntimeError("self.centroids has not yet been set.\nRun Grains().set_centroids() to do so.")
        
        img = copy.deepcopy(self.base_image)
        for i, c in enumerate(self.centroids):
            cv2.circle(img, (c["x"], c["y"]), 3, (0,0,255), thickness=-3)
        cv2.imwrite(self.img_out_fn, img)

    def write_hist_out(self, bins=None):
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

    def _pix_area_to_microns_area(self, target_area_pix):
        image_area_microns = self.height_microns * self.width_microns
        image_area_pix = self.working_image.size
        target_area_microns = (target_area_pix/image_area_pix) * image_area_microns
        return target_area_microns


if __name__ == "__main__":
    g = Grains("test_grains.jpg", 311, 291)
    g.preprocess_image()
    g.set_contours()
    g.set_moments()
    g.set_centroids()
    g.write_img_out()
    g.write_hist_out()
    print("Grain count: {}".format(g.grain_count))
    print("Average grain size: {} um^2".format(g.average_grain_area))