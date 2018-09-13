# Changelog

## 3.7.5

Released Wed September 12, 2018

* Re-wrote reset_weights; just recompiles model
* Fixed error in gridfont loader
* All widgets/pictures are JupyterLab compatible
* Added support for dynamic_pictures on/off; default is off
* SVG arrows are now curves
* New algorithm for bank layout in SVG
* moved dataset.get() to Dataset.get() and net.get_dataset()
* new virtual datasets API, including vmnist, H5Dataset (remote and local)
* better cache in virtual datasets
* Allow dropout to operate on 0, 1, 2, or 3 whole dims
* Added cx.Layer(bidirectional=mode)
* Show network banks as red until compiled
* Rewrote and renamed net.test() to net.evaluate() and net.evaluate_and_label()
	* net.evaluate() for showing results
	* net.evaluate_and_label() for use in plots

## 3.7.4

Released Sun August 19, 2018

* net.pp() gives standard formatting for ints and floats
* Allow negative position in virtual dataset vectors
* Fixed error in colors dataset that truncated the target integer to 8 bits
* Add internal error function to net.compile(error={...})
* New spelling: ConX
* cx.image_to_array() removes alpha
* vshape can be three dimensions (for color images)
* some new image functions: image_resize(), image_remove_alpha()
* renamed "sequence" to "raw" in utils
* Added cx.shape(summary=False), cx.get_ranges(array, form), and get_dim(array, DIMS)
* Use kverbose in train() for all keras activity

## 3.7.3

Released Mon August 13, 2017

* Allow bool values with onehot
* Unfix fixed crossentropy warning
* Allow datasets to be composed of bools
* added temperature to choice()
* Added net.dataset.inputs.test(tolerance=0.2, index=True)

## 3.7.1

Released Fri August 10, 2018

* Separate build/compile --- compile() no longer resets weights;
* added net.add_loss()
* Remove additional_output_banks
* refactor build/compile
* add LambdaLayer with size
* add prop_from_dict[(input, output)] = model

## 3.7.0

Released Tue Aug 7, 2018

* Allow additional output layers for network
* Fix: crossentropy check
* added indentity layer for propagating to input layers
* Include LICENSE.txt file in wheels

## 3.6.10

Released Thu May 17, 2018

* delete layers, IncepetionV3, combine networks
* ability to delete layers
* ability to connect two networks together
* rewrote SVG embedded images to use standard cairosvg
* added inceptionv3 network
* cx.download has new verbose flag
* fixes for minimum and maximum
* import_keras_model now forms proper connections
* Warn when displaying network if not compiled then activations won't be visible
* array_to_image(colormap=) now returns RGBA image

## 3.6.9

Released Fri May 4, 2018

* propagate_to_features() scales layer[0]
* added cx.array
* fixed (somewhat) array_to_image(colormap=)
* added VGG16 and ImageNet notebook
* New Network.info()
* Updated Network.propagate_to_features(), util.image()
* Network.info() describes predefined networks
* new utility image(filename)
* rewrote Network.propagate_to_features() to be faster
* added VGG preprocessing and postprocessing
* Picture autoscales inputs by default
* Add net.picture(minmax=)
* Rebuild models on import_keras
* Added VGG19
* Added vgg16 and idea of applications as Network.networks
* Bug in building intermediate hidden -> output models

## 3.6.7

Released Tue April 17, 2018

* Fixed bug in building hidden -> output intermediate models

## 3.6.6

Released Fri April 13, 2018

* Added cx.view_image_list(pivot) - rotates list and layout
* Added colors dataset
* Added Dataset.delete_bank(), Dataset.append_bank()
* Added Dataset.ITEM[V] = value
	
## 3.6.5

Released Fri April 6, 2018

* Removed examples; use notebooks or help instead
* cx.view_image_list() can have layout=None, (int, None), or (None, int)
* Added cx.scale(vector, range, dtype, truncate)
* Added cx.scatter_images(images, xys) - creates scatter plot of images
* Fixed pca.translate(scale=SCALE) bug
* downgrade tensorflow on readthedocs because memory hog kills build

## 3.6.4

Released Thur April 5, 2018

* changed "not allowed" warning on multi-dim outputs to
  "are you sure?"
* fix colormap on array_to_image; added tests
* fix cx.view(array)
* Allow dataset to load generators, zips, etc.

## 3.6.3

Released Tue April 3, 2018

* Two fixes for array_to_image: div by float; move cmap conversion to end
* Protection for list/array for range and shape
* from kmader: Adding jyro to binder requirements

## 3.6.2

Released Tue March 6, 2018

* added raw=False to conx.utilities image_to_array(), frange(), and reshape()

## 3.6.1

Released Mon March 5, 2018

* SVG Network enhancements
  * vertical and horizontal space
  * fixed network drawing connection paths
* save keras functions
* don't crash in attempting to build propagate_from pathways
* added binary_to_int
* download() can rename file
* fixed mislabeled MNIST image
* better memory management when load cifar
* Network.train(verbose=0) returns proper values
* labels for finger dataset are now strings
* added labels for cx.view_image_list()
* fixed bug in len(dataset.labels)
* added layout to net.plot_layer_weights()
* added ImageLayer(keep_aspect_ratio)
* fixed bugs in datavector.shape
* added dataset.load(generator, count)
* fixed bugs in net.get_weights()/set_weights()
* added network.propagate_to_image(feature=NUM)

## 3.6.0

Released Mon Feb 12, 2018. Initial released version recommended for daily use.

* fixed blurry activation network pictures
* show "[layer(s) not shown]" when Layer(visible=False)
* added fingers dataset
