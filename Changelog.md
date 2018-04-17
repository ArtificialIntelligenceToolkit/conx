# Changelog

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
