# MRAG-KG

I have built 2 knowledge bases generation, given the [Fashionpedia](https://fashionpedia.github.io/home/index.html) dataset.
Here I explain how the datasets are generated.

Scripts used for the 
generation can be found [here](https://github.com/alkidbaci/MRAG-KG).

## About Fashionpedia

Fashionpedia provides the following data that we make use of:

- [Training images](https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip)
- [instances_attributes_train2020](https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json)

The first file is a _zip_ file containing a folder which holds all the images
of fashionpedia dataset.

The second file is a _json_ file that contain the fashionpedia ontology.
In this ontology the main individual is an "annotation" which holds the data for
a certain image. An annotation describes some part of the image (a wearable item)
by specifying the category of the item and the attributes of this item, 
above other informations.

## First Generation

The first generation step consist of creating an rdf ontology/knowledge base 
on the information given in `instances_attributes_train2020.json`. 

A structure of the data is given below:

```
{
 "info": info,
 "categories": [category],
 "attributes": [attribute],
 "images": [image],
 "annotations": [annotation],
 "licenses": [license]
}

info{
  "year" : int,
  "version" : str,
  "description" : str,
  "contributor" : str,
  "url" : str,
  "date_created" : datetime,
}

category{
  "id" : int,
  "name" : str,
  "supercategory" : str,  # parent of this label
  "level": int,           # levels in the taxonomy
  "taxonomy_id": string,
}

attribute{
  "id" : int,
  "name" : str,
  "supercategory" : str,  # parent of this label
  "level": int,           # levels in the taxonomy
  "taxonomy_id": string,
}

image{
  "id" : int,
  "width" : int,
  "height" : int,
  "file_name" : str,
  "license" : int,
  "time_captured": string,
  "original_url": string,
  "isstatic": int, 0: the original_url is not a static url,
  "kaggle_id": str,
}

annotation{
  "id" : int,
  "image_id" : int,
  "category_id" : int,
  "attribute_ids": [int],
  "segmentation" : [polygon] or [rle]
  "bbox" : [x,y,width,height], # int
  "area" : int
  "iscrowd": int (1 or 0)
}
polygon: [x1, y1, x2, y2, ...], where x, y are the coordinates of vertices, int
rle: {"size", (height, widht), "counts": str}

license{
  "id" : int,
  "name" : str,
  "url" : str
}
```

In the script named `first_generation.py` I use [rdflib](https://rdflib.readthedocs.io/en/stable/) to create a graph which
I populate by adding axioms via the same libray.

1. First I add a class for each of the following items:
"info", "category", "attribute", "image", "annotation" and "license".
2. Then I add object properties for connections that are done using "id".
  For example an annotation has an `"image_id"` which is referring to the 
  image it belongs. Therefore, for the class annotation I will create an
  object property `"hasImage"`. The same is done for each id connected
  entity.
3. For the rest of the data that an entry has, I create a datatype property to
   represent them in the knowledge base. 
4. The last step consist of adding the individuals by going through each entry 
   in the dataset and adding the respective classes and properties to it.

By the end of the 4th step, the first knowledge base generation will be completed.

## Second Generation

For the second knowledge base generation, `second_generation.py` script is used.

For the second generation I want the only individuals to be images. Therefore,
I have only one class, which is `Image`. 
These image individuals contain all the information from the annotations
belonging to that image.

That means that an image can contain more than one wearable items that is described
by an annotation. So basically I have merged together all the information there is 
for an image.

There are only data properties on this dataset, no object properties, 
because we only describe data for images and there is no need to have a 
relation between these images.

In this generation I have included only the necessary information and omitted the 
rest. All the information for an annotation that belongs to the image is concluded
in a string and added as a data property to the image.

A structure of the data is given below:

```
image{
  "file_name" : str,
  "width" : int,
  "height" : int,
  "descriptions": { 
    "desc1": str,
    "desc2": str,
    ... 
  }
}
```

This is more of a json description for the sake of understanding, but the real
data is in XML/RDF format. Each annotation is represented
by a "has_description" property or by "desc1", "desc2", "..." in the example above.

I have not included `original_url` as a property because that usually refers
to the website that hosts the image and not direct link to the image itself, 
so basically its trivial information.

