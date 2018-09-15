# Hidden Dimensions

## Intro

**Goal**: Explore properties of Latent spaces to facilitate domain knowledge extraction in unsupervised/semi-supervised set-ups.

**Data domains**: Text and Image. Later on Graphs.

**Applications**: 

- *By discovering how to create Well-Clustered latent spaces, we can enable the extraction of Domain Constructs and Domain Concepts.*
- **Examples**:
    - Vector Directions in Word Embeddings
    - Arithmetic on Image Embeddings
- *By designing Algorithms for Navigation in a Meaningful latent space, we can imitate a generative process which might resemble abstract reasoning.*
- **Examples**: 
    - Extracting hierarchies and relationships between objects from an image through traversal of sub-images directly in the latent space (**Image embeddings**);
    - Writing code on top of pre-defined functionality by expanding parse trees directly in the latent space (**Graph embeddings**);
- *The analogy to the human mind is that latent representations are thoughts derived from perceptual information.*
- *The navigation in the latent space then corresponds to thinking about thoughts, a mechanism which allows us to generate relevant actions which derive from an abstract representation of a situation, rather than exact copies of actions performed in the past in a similar situation.*

**Applications linked to our past research**:

- *Clustering Computer Programs based on Spatio-Temporal Features*
- *Aster Project: AST derived Code Representations for General Code Evaluation and Generation*
- *Clustering and Visualization of Latent Representations*
- *Learning Semantic Web Categories by Clustering and Embedding Web Elements*

## Representations


| Representation | Description |
|:--------------:|:-----------:|
|![Autoencoder](https://upload.wikimedia.org/wikipedia/commons/2/28/Autoencoder_structure.png "Autoencoder") Wikipedia, Autoencoder| The latent representation in a typical autoencoder is a kind of black box or bottleneck inside a system that optimizes the information compression of the input data constrained by minimizing the data reconstruction error.|

### Basics of Variational Auto-encoder

**Idea**: Model latent variable as a variable drawn from a learned probability distribution.

| Representation | Description |
|:---:|:---:|
| ![VAE Encoder Decoder](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/encoder_decoder.png "VAE Encoder Decoder")  from *https://jaan.io/what-is-variational-autoencoder-vae-tutorial/*| **Encoder**: q models probability of hidden variable given data, **Decoder**: p models data probability given hidden variable |

| ![VAE Derivation](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/vae_1.jpg "VAE Derivation") | ![VAE Properties](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/vae_2.jpg "VAE Properties") |
|:---:|:---:|
| **VAE Derivation**:  | **VAE Properties**: |
| Loss function consists of 2 terms - **Reconstruction Error**: how well samples are reconstructed from hidden variable, **KL Divergence**: penalizes data points from occupying distant regions in the latent space.| Encoding sample is deterministic. Then z is drawn from probability distribution q(z \| x). Reconstructed \~x is also drawn from probability distribution p(x \| z).|


## Properties

| ![Word Directions](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/word_directions.png "Word Directions") | ![Image Arithmetic](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/image_arithmetic.png "Image Arithmetic") |
|:---:|:---:|
| **Word2Vec** (Mikolov et al. (2013)) | **DCGAN** (Radford et al. (2015)) |


### Samples Density and Principal Component Analysis

![Clustering vs. PCA](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/mnist_clust_vs_pca.png "Sample 4")

**Observations regarding data complexity**:

- only k principle components obtained from Z (latent space) will be meaningful (in the case of MNIST k between 10 and 20)

### Non-Linear Projections, Sub-Spaces and Clustering

**Application to semi-supervised/few-shots learning**:

- convex hull defined by few labeled shots on clusters formed from latent representations

### Navigation in Latent Spaces and Attractors

| ![Latent Arithmetic](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/latent_arithmetic.jpg "Latent Arithmetic") | ![Latent Navigation](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/latent_navigation.jpg "Latent Navigation") |
|:---:|:---:|
| **Latent Arithmetic**: Do sub-images form hierarchical clusters or separate clusters in the latent space? Can header and container be added (with a learnt operator, in the latent representation) to output the latent resulting page? Can algorithms for navigation in the tree structure be directly implemented in the latent space? | **Latent Navigation**: Linear interpolation between Z of samples; Clock-wise rotation (interpolation on a curved surface); Valley/ridge navigation on projected surface of latent space - valleys/ridges can be found by clustering: dense regions form valleys, while sparse regions determine ridges (topology map). |

## Tasks

### Multi-Label Figures with Shape & Color

**Data domain**: synthetic, image

**Dataset**:

- 28 x 28 x 3 sized images
- 2 labels: shape and color
- shapes: square, circle, triangle
- colors: red, green, blue

**Primary Goal**:

- explore multi-label, multi-class classification problem
- 3 options: 
	- multi-label (multiple logit sets)
	- combination of labels (one logit set, each logit is a combo)
	- one neural-network for each label

**Secondary Goal**:

- inspect learned features in intermediate CNN layers

| Sample | Description | Sample | Description |
|:------:|:-----------:|:------:|:-----------:|
|![Sample 1](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/MLSC/373_square_red.png "Sample 1")| red square |![Sample 5](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/MLSC/560_triangle_red.png "Sample 5")| red triangle |
|![Sample 2](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/MLSC/402_circle_blue.png "Sample 2")| blue circle |![Sample 6](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/MLSC/573_circle_red.png "Sample 6")| red circle |
|![Sample 3](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/MLSC/418_square_green.png "Sample 3")| green square |![Sample 7](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/MLSC/569_triangle_blue.png "Sample 7")| blue triangle |
|![Sample 4](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/MLSC/500_triangle_green.png "Sample 4")| green triangle |![Sample 8](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/MLSC/598_square_blue.png "Sample 8")| blue square |

|Primary Results|Primary Conclusions|
|:------------------:|:----------------------:|
|![Multi-Label vs. Combo](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/multi_label_vs_combo.png "Multi-Label vs. Combo")|Multi-label performs slightly better than combinations of all labels. The total number of logits add in the case of multi-label, but multiplies in the case of combinations.|

### Spatial Relationes between Figures and Scene Description

**Data domain**: synthetic, image and text

**Dataset**:

- 56 x 56 x 3 sized images
- 2 figures per image, obtained by concatenation of shape-color figures
- 2 relations: *"above"* and *"next_to"*

| Sample | Description |
|:------:|:-----------:|
|![Sample 1](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/SRSC/45_green_square_above_blue_square.png "Sample 1")| green square above blue square|
|![Sample 2](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/SRSC/85_green_triangle_above_red_square.png "Sample 2")| green triangle above red square|
|![Sample 3](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/SRSC/87_blue_circle_next_to_red_square.png "Sample 3")| blue circle next_to red square|
|![Sample 4](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/SRSC/148_green_square_next_to_blue_triangle.png "Sample 4")| green square next_to blue triangle|

**Extension**: Latent arithmetic with figures

### Synthetic Webpages

**Data domain**: synthetic, image, tree and text

**Extension**: Latent arithmetic with figures

### MNIST

**Data domain**: natural, image

**Primary Idea**: Clutering of latent representation


**Secondary Idea**: Latent interpolation

**Third Idea**: Few-shots learning after unsupervised training

### Sort of CLEVR

**Data domain**: synthetic, image and text

**Purpose**: Test mock models for CLEVR with less data.

### CLEVR

**Data domain**: natural, image and text

[Dataset Link](https://cs.stanford.edu/people/jcjohns/clevr/)

| Sample | Description |
|:------:|:-----------:|
|![CLEVR Sample](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/clevr_sample.png "CLEVR Sample")|3D objects rendered in a scene. Questions test spatial reasoning.|
|**Questions**|**Answers**|
|Are there an **equal number** of large things and metal spheres?|Yes|
|What **size** is the cylinder that **is left** of the brown metal thing that is left of the **big sphere**?|Large|

**Premise**: Question types include attribute identification, counting, comparison, spatial relationships, and logical operations.

**Primary Question**: Can we group similar questions together?

**Secondary Question**: Can we generate new questions that are relevant?

The dataset is a great subject for the application of **Relational Networks** [G1], which belong to the family of **Graph Nets**. 

**Tutorials**:

- [Tutorial 1](https://hackernoon.com/deepmind-relational-networks-demystified-b593e408b643)
- [Tutorial 2](https://rasmusbergpalm.github.io/recurrent-relational-networks/)

#### Relation to Code Generation

Models that work well for this dataset could be extended to do some sort of programming through the following links.

*Primary Idea*: **Question asking**

- When writing code, programmers often ask questions about the state of the program and from the answers infer the next steps to take in order to accomplish their goal. 
- For instance, in the world of objects, we might ask which figures to swap such that a certain order relation is satisfied.

*Topics to explore further:* which questions do you ask yourself when writing code/solutions to programming tasks?

*Secondary Idea*: **Tangible programming**

- A more concrete example would be to group objects in a sequence such that similar colors are consecutive.
- This would translate into asking a lot of questions similar to the ones showcased by the CLEVR dataset

*Topics to explore further:* tangible interfaces, embodied cognition and interaction.

*Third Idea*: **Logical questions modeled through programming**

- Does the first object have the same color as the second one?
- More generally, does the object on position i have the same color as the object in position i + 1
- Where is the first object to have red color?
- Are there any objects colored with red at all?

*Fourth Idea*: **Causal inference from answers to logical questions**

- If object on third position is yellow and object on sixth position is blue, swapping the two will result in which color on 6th position?
- Does the initial color on 6h position even matter?
- What if a pattern of movements always results the same output?
- How can such patterns be found? 
- Can an agent model optimal behavior and generate such patterns?
- Does curiosity help?

## References

[G1] [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf), A. Santoro et al, 2017