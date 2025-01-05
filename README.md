# Multimodal Retrieval-Augmented Generation over Knowledge Graph

In this repository you will find all the scripts used during the writing of the thesis.

To reproduce the results and necessary data structures only the `general_working_directory` should be considered. The other directories
can be ignored.

Prerequisites before reproducing the kg generation and evaluation results:

Install python v3.10.13 or later.

Clone the repository, create a virtual environment and install dependencies:

```commandline
# 1. clone 
git clone https://github.com/dice-group/MRAG-KG.git
# 2. setup virtual environment
python -m venv venv 
# 3. activate the virtual environment
source venv/bin/activate # for Unix and macOS
.\venv\Scripts\activate  # for Windows
# 4. install dependencies
pip install -r requirements.txt
```

Move to the general working directory:

```commandline
cd general_working_directory
```


## 1. KG generation
We have generated 3 knowledge graphs, given the [Fashionpedia](https://fashionpedia.github.io/home/index.html) dataset.
Here we explain how the KGs are generated.

### About Fashionpedia

Fashionpedia provides the following data that we make use of:

- [Training images](https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip)
- [instances_attributes_train2020](https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json)

The first file is a _zip_ file containing a folder which holds all the images
of fashionpedia dataset.

The second file is a _json_ file that contain the fashionpedia ontology.
In this ontology the main individual is an "annotation" which holds the data for
a certain image. An annotation describes some part of the image (a garment/garment part)
by specifying the category of the garment and its attributes, above other information.

### First Generation

The first generation step consist of creating an RDF KG which representing the data given in `instances_attributes_train2020.json`. 

The structure of the fashionpedia ontology is given below:

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

In the script named `first_generation.py` we use [rdflib](https://rdflib.readthedocs.io/en/stable/) to create a graph which
we populate by adding axioms via the same libray.

1. First we add a class for each of the following items:
"info", "category", "attribute", "image", "annotation" and "license".
2. Then we add object properties for connections that are done using "id".
  For example an annotation has an `"image_id"` which is referring to the 
  image it belongs. Therefore, for the class annotation we will create an
  object property `"hasImage"`. The same is done for each id connected
  entity.
3. For the rest of the data that an entry has, we create a datatype property to
   represent them in the knowledge base. 
4. The last step consist of adding the individuals by going through each entry 
   in the dataset and adding the respective classes and properties to it.

By the end of the 4th step, the first knowledge base generation will be completed.

### Second Generation

For the second knowledge base generation, `second_generation.py` script is used.

For the second generation we want the only individuals to be images. Therefore,
we have only one class, which is `Image`. 
These image individuals contain all the information from the annotations
belonging to that image.

That means that an image can contain more than one wearable items that is described
by an annotation. So basically, we have merged together all the information there is 
for an image.

There are only data properties on this dataset, no object properties, 
because we only describe data for images and there is no need to have a 
relation between these images.

In this generation we have included only the apparel-descriptive information and omitted the 
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

For the sake of understanding we are showing this in a json format, but this data exist only
in RDF/XML format. Each annotation is represented  by a "has_description" property denoted as "desc1", "desc2", "..." 
in the example above.

We have not included `original_url` as a property because that usually refers
to the website that hosts the image and not direct link to the image itself, 
so basically its trivial information.


### Third Generation
We use a LLM to generate a short description about each image in the dataset. This description is then added to each 
instance in the second KG using a data property.

### Reproducing steps: commands and order of execution

_* The service for the LLM required before running the following scripts._

1. `python first_kg_generation.py` &rarr; Generated the first KG (From JSON to RDF/XML)
2. `python second_kg_generation.py` &rarr; Generated the second KG (Subgraph Summarization)
3. `python third_kg_generation.py` &rarr; Generated the third KG (Enrichment with Multimodal
LLM-generated Context)
4. `python second_kg_generation_v2.py` &rarr; Generate the second KG v2 (KG for CEL)


## Benchmark generation
We crate two benchmark datasets. The goal is to represent each instance (image) of KG by a question/query generated by an
LLM. This benchmark makes it possible to evaluate the retrival models where for an asked question we expect the relevant 
instance as specified in the benchmark.

The second benchmark is restructured to group similar question together by finding k-nearest neighbor for each of them.
Each instance in the fist benchmark now can be mapped to the k-nearest neighbor instances in terms of question similarity,
expanding the set of relevant instances for a given question.

### Reproducing steps: commands and order of execution

_* Services for the LLM and embedding model required before running the following scripts._

1. `python question_generation.py` &rarr; Generates a question for each instance and stores the data in `questions.json`.
2. `python 1st_benchmark_generation.py` &rarr; Generates multiple simpler questions based on the single question per instance that was generated in the previous step. Result is the first benchmark dataset (`first_benchmark.json`) where each 
instance is mapped to a string that contains multiple questions divided by `*` and sometimes divided by numerical values (enumerated).
3. `python 2nd_benchmark_generation.py` &rarr; For each instance in the first benchmark select a random question and generate embedding, find k-nearest neighbor using embeddings and store them for each instance. Results is the 
second benchmark dataset (`second_benchmark.csv`). The embeddings of questions are also stored (filename: `question_embeddings.csv`).


## 2. Ranking made by the first retrieval method

We test 2 retrieval models in our first retrieval method, the embedding-based retriever and the BM25 retriever.
For the embedding-based retriever we first need to generate embeddings for the documents. A document is the concatenation of
all descriptions for each instance in the KG. We generate embeddings for documents in the second KG and documents in the third KG.

### Reproducing steps: commands and order of execution

_* The service for embedding model required before running the following scripts._

1. `python docs_embedding_generation.py -kg_path fashionpedia-second-generation.owl`
2. `python docs_embedding_generation.py -kg_path fashionpedia-third-generation.owl`
3. `python embedding-based_retriever_evaluation_b1.py -embeddings embeddings_second_kg.csv`
4. `python embedding-based_retriever_evaluation_b1.py -embeddings embeddings_third_kg.csv`
5. `python embedding-based_retriever_evaluation_b2.py -embeddings embeddings_second_kg.csv`
6. `python embedding-based_retriever_evaluation_b2.py -embeddings embeddings_third_kg.csv`
7. `python bm25_evaluation_b1.py -kg_path fashionpedia-second-generation.owl`
8. `python bm25_evaluation_b1.py -kg_path fashionpedia-third-generation.owl`
9. `python bm25_evaluation_b2.py -kg_path fashionpedia-second-generation.owl`
10. `python bm25_evaluation_b2.py -kg_path fashionpedia-third-generation.owl`


## 3. Ranking made by the second retrieval method

In the second retrieval method we perform a document re-ranking after initially ranking them using the 
embedding-based retrieval method. The re-raking process includes learning a class expression using
a CEL model and classifying instances on the KG. The top k documents of the classified instances are used
to generate a summary that will be encoded into a vector space and used to rank the classified instances
based on cosine-similarity of the summary and the classified documents where a higher similarity score indicates
a higher relevance.


### Reproducing steps: commands and order of execution

1. `python 2nd_retrieval_method_evaluation.py -lp_size 40 -kg_path fashionpedia-second-generation.owl`
2. `python 2nd_retrieval_method_evaluation.py -lp_size 20 -kg_path fashionpedia-second-generation.owl`
3. `python 2nd_retrieval_method_evaluation.py -lp_size 10 -kg_path fashionpedia-second-generation.owl`
4. `python 2nd_retrieval_method_evaluation.py -lp_size 40 -kg_path fashionpedia-third-generation.owl`
5. `python 2nd_retrieval_method_evaluation.py -lp_size 20 -kg_path fashionpedia-third-generation.owl`
6. `python 2nd_retrieval_method_evaluation.py -lp_size 10 -kg_path fashionpedia-third-generation.owl`


## 4. MRR and Hits@K

After generating all the ranking files, we use a single script to calculate and print the MRR and Hits@k for k in {10,20,50,100}.

```commandline
python mrr_and_hits@k.py
```

## 5. CEL models evaluation

We also evaluate the performance of 3 CEL models from Ontolearn in classifying 
clusters of instances (identified by KNN algorithm - second benchmark).

### Reproducing steps: commands and order of execution

1. `python cel_evaluation.py -model tdl`
2. `python cel_evaluation.py -model celoe`
3. `python cel_evaluation.py -model drill`


## 6. TSNE plot

The TSNE plot shows clusters created by the evaluation sample instances.
To reproduce the plot use the following command:

```commandline
python TSNE_plot.py
```

