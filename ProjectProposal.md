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
- *Aster Project*
- *Learning Semantic Web Categories by Clustering and Embedding Web Elements*

## Representations

The latent representation in a typical autoencoder is a kind of black box or bottleneck inside a system that optimizes the information compression of the input data constrained by minimizing the data reconstruction error.

![Autoencoder](https://upload.wikimedia.org/wikipedia/commons/2/28/Autoencoder_structure.png "Autoencoder")

Wikipedia, Autoencoder

### Basics of Variational Auto-encoder


| ![VAE Derivation](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/vae_1.jpg "VAE Derivation") | ![VAE Properties](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/vae_2.jpg "VAE Properties") |
|:---:|:---:|
| **VAE Derivation**:  | **VAE Properties**: |


## Properties

| ![Word Directions](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/word_directions.png "Word Directions") | ![Image Arithmetic](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/image_arithmetic.png "Image Arithmetic") |
|:---:|:---:|
| **Word2Vec** (Mikolov et al. (2013)) | **DCGAN** (Radford et al. (2015)) |


### Samples Density and Principal Component Analysis

### Non-Linear Projections, Sub-Spaces and Clustering

### Navigation in Latent Spaces and Attractors

| ![Latent Arithmetic](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/latent_arithmetic.jpg "Latent Arithmetic") | ![Latent Navigation](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/latent_navigation.jpg "Latent Navigation") |
|:---:|:---:|
| **Latent Arithmetic**: Do sub-images form hierarchical clusters or separate clusters in the latent space? Can header and container be added (with a learnt operator, in the latent representation) to output the latent resulting page? Can algorithms for navigation in the tree structure be directly implemented in the latent space? | **Latent Navigation**: Linear interpolation between Z of samples; Clock-wise rotation (interpolation on a curved surface); Valley/ridge navigation on projected surface of latent space - valleys/ridges can be found by clustering: dense regions form valleys, while sparse regions determine ridges (topology map). |
