## Requirements
Used libraries
* keras-gpu = 2.2.4 
* tensorflow-gpu = 1.15.0
* scikit-image = 0.16.2
* opencv-python = 4.2.0
* maptplotlib = 3.1.3
* numpy = 1.18.1

For develop library used 
* Anaconda environment
* Python 3.7
* CUDA 


## Samples of use

### Search crowns with use Watershed algorithm

    from watershed_prediction import predict_big_image
    from skimage import io
    from matplotlib.pyplot as plot
    import time
    count_trees, image = predict_big_image("path//to//imame")
    print("Find ", count_trees, " trees")
    plot.imshow(image)
    plot.show()
    
### Result

![Result work of watershed algorithm](/gallery/watershed_crowns_delination_result.png)
    
### Search crowns with use Mask R-CNN neural network

    from mrcnn_prediction import predict_big_image
    from skimage import io
    from matplotlib.pyplot as plot#
    path_weights = "my_network_shapes.h5"
    count_trees, image = predict_big_image(path_weights, path_image, 360)
    print("Find ", count_trees, " trees")
    plot.imshow(image)
    plot.show()
    
### Result

![Result prediction Mask R-CNN](/gallery/using_mask_rcnn_with_default_parameters.png)

## Settings

###Setting of watershed algorithm

Using default parameters for Watershed algorithm

    count_trees, image = predict_small_image(path//to//image)
    
![Using watershed with default parameters](/gallery/using_watershed_with_default_params.png)

Blur reduction away bilateral filter

    count_trees, image = predict_small_image(path//to//image, d=1, sigmaColor=10, sigmaSpace=10)
    
![Using watershed with blur reduction](/gallery/using_watershed_with_blur_reduction.png)
    
Blur increase away bilateral filter

    count_trees, image = predict_small_image(path//to//image,  d=20, sigmaColor=150, sigmaSpace=150)
    
![Using watershed with blur increase](/gallery/using_watershed_with_blur_increase.png)
    
Reduce the min area parameter of crowns filter

    count_trees, image = predict_small_image(path//to//image, min_area=20)
    
![Reduce min area parameter](/gallery/using_watershed_with_reduce_min_area.png)


Reduce the footprintSize parameter setting size of region for search local maximum

    count_trees, image = predict_small_image(path//to//image, footprintSize=(12, 12))

![Reduce the footprintSize parameter](/gallery/using_watershed_wih_reduce_footprint_size.png)

Increase the footprintSize parameter setting size of region for search local maximum

     count_trees, image = predict_small_image(path//to//image, footprintSize=(70, 70))
     
![Increase the footprintSize parameter](/gallery/using_watershed_with_increase_footprint_size.png)


###Setting of Mask R-CNN prediction

Using default parameters

    count_trees, image = predict_big_image(path//to/weights, path//to//image)

![Result prediction Mask R-CNN with default parameters](/gallery/using_mask_rcnn_with_default_parameters.png)

Increase size of split images

    count_trees, image = predict_big_image(path//to/weights, path//to//image, size_of_slice=600)
    
![Result prediction Mask R-CNN with increase size of split image](/gallery/using_mask_rccn_with_reduce_size_of_split_image.png)

Reduce size of split images

    count_trees, image = predict_big_image(path//to/weights, path//to//image, size_of_slice=250)
    
![Result prediction Mask R-CNN with reduce size of split image](/gallery/using_mask_rccn_with_increase_size_of_split_image.png)