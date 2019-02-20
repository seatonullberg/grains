import grains

if __name__ == "__main__":
    # in this example the width and height are not to scale
    analyzer = grains.analyzer.GrainsAnalyzer(input_fn="test_grains.jpg", 
                                              height_microns=311, 
                                              width_microns=291)
    
    analyzer.preprocess_image()  # do thresholding and noise reduction
    analyzer.set_contours()      # find contours
    analyzer.set_moments()       # find moments from those contours
    analyzer.set_centroids()     # find centroids from those moments
    analyzer.write_summary()     # write basic text file summary of the findings
    
    # write the image with red dots on each centroid
    grains.plotting.write_image(img=analyzer.base_image,
                                centroids=analyzer.centroids,
                                filename="test_grains.image.png")
    
    # write a histogram of the grain area distribution
    grains.plotting.write_histogram(data=analyzer.contour_areas_microns(),
                                    filename="test_grains.histogram.png")