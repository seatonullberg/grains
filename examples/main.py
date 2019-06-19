import grains


if __name__ == "__main__":
    # in this example the width and height are not to scale
    ga = grains.analysis.GrainsAnalyzer(input_fn="test_grains.jpg", 
                                        height_microns=311, 
                                        width_microns=291)
    
    ga.preprocess_image()  # do thresholding and noise reduction
    ga.set_contours()      # find contours
    ga.set_moments()       # find moments from those contours
    ga.set_centroids()     # find centroids from those moments
    ga.write_summary("test_grains.summary.txt")  # write basic text file summary of the findings
    
    # write image with red dots superimposed over each detected centroid (grain center)
    grains.graphics.write_centroids_image(img=ga.base_image,
                                          centroids=ga.centroids,
                                          filename="test_grains.centroids.png")
    
    # write a histogram of the grain area distribution
    grains.graphics.write_histogram(data=ga.areas,
                                    filename="test_grains.histogram.png")
