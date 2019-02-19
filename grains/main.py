import cv2
import numpy as np

class Grains(object):

    def __init__(self, input_fn):
        self.input_fn = input_fn
        self.img_out_fn = "{}.grains.image.png".format(self.input_fn)
        self.plot_out_fn = "{}.grains.plot.png".format(self.input_fn)
        self.stats_out_fn = "{}.grains.stats.txt".format(self.input_fn)

    @property
    def contours(self):
        img = cv2.imread(self.input_fn)
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _ret, threshold = cv2.threshold(grayscale_img, 50, 255, 0)
        kernel = np.ones((3,3), np.uint8)
        opened_img = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        cv2.imshow("test", opened_img)
        cv2.waitKey(0)
        conts, _hierarchy = cv2.findContours(opened_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return conts

    @property
    def moments(self):
        moms = []
        for c in self.contours:
            m = cv2.moments(c)
            moms.append(m)
        return moms

    @property
    def centroids(self):
        cntrs = []
        for m in self.moments:
            coords = {}
            if m["m00"] == 0:
                coords["x"] = 0
                coords["y"] = 0
            else:
                coords["x"] = int(m["m10"] / m["m00"])
                coords["y"] = int(m["m01"] / m["m00"])
            cntrs.append(coords)
        return cntrs             
            
    def write_image(self):
        img = cv2.imread(self.input_fn)
        for i, c in enumerate(self.centroids):
            cv2.putText(img, "{}".format(i), (c["x"], c["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite(self.img_out_fn, img)
        

if __name__ == "__main__":
    g = Grains("test_grains.jpg")
    g.write_image()