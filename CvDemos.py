from Utilities import DiagnoseTool


def cv_basic_test():
    from Cv.Basics.ChannelsMerge import merge_test
    from Cv.Basics.ChannelsSplits import split_test
    from Cv.Basics.ConcatenateImages import concatenation_test
    from Cv.Basics.CropAnImage import cropping_test
    from Cv.Basics.Transforms import show_rotated_images, show_reshaped_images

    from Cv.Basics.UseCamera import call_camera

    img_color = DiagnoseTool.load_image_color("Data/Illustrations/kono_sekai.jpg")
    img_gray = DiagnoseTool.load_image_gray("Data/Illustrations/opencv_logo.jpeg")

    # do images combination
    merge_test(img_color, img_gray)

    # do image splitting
    split_test(img_color)

    # do concatenation test
    concatenation_test(img_gray)

    # cropping test
    cropping_test(img_color)

    # transformed and reshaped images
    show_reshaped_images(img_gray)
    show_rotated_images(img_gray)

    # call camera
    call_camera()


def cv_gray_histogram_test():
    from Cv.GrayHistogram.IntensityTransformation import intensity_test_1
    from Cv.GrayHistogram.IntensityTransformation2 import intensity_test_2
    from Cv.GrayHistogram.RGBConvert2Grayscale import rgb_convert_test

    # intensity transform
    intensity_test_1()
    intensity_test_2()

    # rgb to gray
    rgb_convert_test()


def cv_noise_test():
    from Cv.Noise.NoiseGenerator import noise_generator_test
    from Cv.Noise.PoissonNoise import poisson_noise_test

    # salt pepper and gaussian
    noise_generator_test()

    # poisson
    poisson_noise_test()


def cv_spatial_filters():
    from Cv.Spatial.SpatialFiltering1 import median_filter_test
    from Cv.Spatial.SpatialFiltering2 import laplacian_op_test

    median_filter_test()
    laplacian_op_test()


def cv_image_deterioration():
    from Cv.Degradation.SpatialNoise import spatial_noise_analysis
    from Cv.Degradation.FrequencyNoise import frequency_noise_analysis

    # image spatial noise deterioration
    spatial_noise_analysis()

    # image frequency noise deterioration
    frequency_noise_analysis()


if __name__ == "__main__":
    # basic operations
    cv_basic_test()

    # histogram
    cv_gray_histogram_test()

    # noise
    cv_noise_test()

    # spatial filters
    cv_spatial_filters()
