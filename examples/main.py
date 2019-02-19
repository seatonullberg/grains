import grains

if __name__ == "__main__":
    # in this example the width and height are not to scale
    g = grains.GrainsAnalyzer("test_grains.jpg", 311, 291)
    # do thresholding and noise reduction
    g.preprocess_image()
    # find contours
    g.set_contours()
    # find moments from those contours
    g.set_moments()
    # find centroids from those moments
    g.set_centroids()
    # write the image with red dots on each centroid
    g.write_img_out()
    # write a histogram of the grain area distribution
    g.write_hist_out()
    # write basic text file summary of the findings
    g.write_stats_out()