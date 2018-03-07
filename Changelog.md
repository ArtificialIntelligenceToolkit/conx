# Changelog

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
