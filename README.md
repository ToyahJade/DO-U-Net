# Duel-Output U-Net (DO-U-Net) for Segmentation and Counting 

DO-U-Net is a Neural Network approach for the segmentation and counting of objects in images. The network produces two output masks: one showing the edges of the objects, and a second segmentation mask as you would normally obtain from the traditional [U-Net](https://doi.org/10.1007/978-3-319-24574-4_28). These are then combined to produce a final segmentation mask in which closely co-located or overlapping objects are distinct from one another, thus allowing the segmented objects to be counted. 

<img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/tents_final_mask.jpg" height="300" title="Example output mask for satellite imagery"> <img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/blood_final_mask.jpg" height="300" title="Example output mask for a blood smear image">

DO-U-Net is an effective approach for when the size of an object needs to be known, as well as the number of objects in the image. DO-U-Net segments closely co-located and overlapping objects successfully, making it a worthy choice when objects are close by and/or overlapping in images. By segmenting images, we are also able to know the exact location of the segmented objects, which can prove extremely useful when looking at, for example, satellite imagery.

The open-access paper which first described DO-U-Net can be found [here](https://doi.org/10.1007/978-3-030-44584-3_31). See also the [IDA 2020 conference website](https://ida2020.org/).

## Introduction 

DO-U-Net was initially created to segment and count Internally Displaced People (IDP) camps in Afghanistan. This work was requested by the Norwegian Refugee Council (NRC), who are interested in the number, flow and concentration of IDP camps so that they can provide the most effective aid possible. There are over 3 million IDP’s in Afghanistan alone, and over 40 million worldwide. 

The NRC needed to know how many tents existed in each IDP camp, as well as the size and exact location of the tents. Whilst other segmentation methodologies, including the standard U-Net, struggled to distinguish between closely co-located tents, the DO-U-Net has no issue with segmenting these as separate objects. 

To show that DO-U-Net works in other environments, we also tested it on blood smear images from the [Acute Lymphoblastic Leukemia (ALL) Image Database for Image Processing](https://homes.di.unimi.it/scotti/all/). The markup data included in this repository corresponds to these images. The images themselves, however, are not included due to their ownership and sharing permissions.

## Training Data

The training data takes the form of polygons, which are stored in a json file. Example files for the blood smear images can be found in [data](https://github.com/ToyahJade/DO-U-Net/tree/master/data), which has been spit into train and test folders. These markup files were made using a custom GUI. 

<img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/tents_2_clip.png" height="200" title="Example satellite imagery clip"> <img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/tents_2_clip_markup.png" height="200" title="Example markup of satellite imagery clip"> <img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/blood_2_clip.png" height="200" title="Example blood smear image clip"> <img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/blood_2_clip_markup.png" height="200" title="Example markup of blood smear image clip">

## DO-U-Net Structure

Two versions of the DO-U-Net exist: a simplified version, and a deeper “Scale-Invariant” version. 

### Traditional DO-U-Net

<img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/do-u-net.jpg" height="200" title="DO-U-Net structure">

The traditional DO-U-Net was designed such that it could be run on laptops with lower-range GPU’s, making it more accessible to the charities which can benefit from it. We also do not feed in a full image for training, but instead a chip of the image. This means the memory requirements of the computer is also reduced, thus retaining the accessibility we have strived for. 

### Scale-Invariant DO-U-Net

<img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/do-u-net_scale-invariant.jpg" height="200" title="Scale-Invariant DO-U-Net structure">

The Scale-Invariant DO-U-Net is a deeper version of the DO-U-Net which we previously outlined. This means the computational requirements are slightly increased, but the network works better over images containing wildly differing scales of objects. 

## Output Masks 

The two output masks produced by the DO-U-Net represent the edges of the objects in the image, and the segmentations of said objects. 

<img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/tents_edge_mask.jpg" height="150" title="Example edge mask for satellite imagery, overlayed on imagery"> <img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/tents_segmentation_mask.jpg" height="150" title="Example segmentation mask for satellite imagery, overlayed on imagery"> <img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/blood_edge_mask.jpg" height="150" title="Example edge mask for blood smear image, overlayed on image"> <img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/blood_segmentation_mask.jpg" height="150" title="Example segmentation mask for blood smear image, overlayed on image">

By simply subtracting the edge mask from the segmentation mask, we produce a final segmentation mask containing no overlaps of images. 

<img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/tents_final_mask.jpg" height="300" title="Example final output mask for satellite imagery, overlayed on imagery"> <img src="https://raw.githubusercontent.com/ToyahJade/DO-U-Net/master/images/blood_final_mask.jpg" height="300" title="Example final output mask for blood smear image, overlayed on image">


## Counting of Objects

The objects can be counted since the masks produced contain no overlaps. We applied a threshold to remove negative values from the final masks, which may occur due to the subtraction. We then applied the [Marching Squares Algorithm](https://scikit-image.org/docs/0.5/auto_examples/plot_contours.html). 

This methodology allows the mask to be converted back into a json. Any corrections required can then be made using our markup GUI, thus allowing any false positives to be removed and any false negatives to be rectified. This corrected json can then be used in further training of the model, if required.  

# Contact
For more information, please contact [toyah.overton@brunel.ac.uk](mailto:toyah.overton@brunel.ac.uk)
