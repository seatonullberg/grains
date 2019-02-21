import os
import click
import grains


@click.command()
@click.argument("filename")
@click.option("--h", default=0, help="content height in microns")
@click.option("--w", default=0, help="content width in microns")
@click.option("--no_histogram", is_flag=True, help="skip histogram plot generation")
@click.option("--no_centroids", is_flag=True, help="skip image of superimposed centroids generation")
@click.option("--no_summary", is_flag=True, help="skip text summarization generation")
def cli(filename, h, w, no_histogram, no_centroids, no_summary):
    # verify arguments
    if h == 0 or w == 0:
        click.echo("Values for height (--h) and width (--w) must be provided for analysis.")
        exit()

    # init analyzer
    ga = grains.analysis.GrainsAnalyzer(input_fn=filename,
                                        height_microns=h,
                                        width_microns=w)
    ga.preprocess_image()
    ga.set_contours()
    ga.set_moments()
    ga.set_centroids()

    filename = os.path.basename(filename)
    extensionless = "".join(filename.split(".")[:-1])  # generate a base filename
    # check flags
    if not no_histogram:
        fn = "{}.histogram.png".format(extensionless)
        grains.graphics.write_histogram(data=ga.areas,
                                        filename=fn)
    if not no_centroids:
        fn = "{}.centroids.png".format(extensionless)
        grains.graphics.write_centroids_image(img=ga.base_image,
                                              centroids=ga.centroids,
                                              filename=fn)
    if not no_summary:
        fn = "{}.summary.txt".format(extensionless)
        ga.write_summary(filename=fn)
