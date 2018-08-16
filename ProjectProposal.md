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
    - Writing code on top of pre-defined functionality by expanding parse trees directly in the latent space (**Graph embeddings**);
    - Extracting hierarchies and relationships between objects from an image through traversal of sub-images directly in the latent space (**Image embeddings**);
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

## Properties

**Word2Vec**

![Word Directions](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/word_directions.png "Word Directions")

Mikolov et al. (2013)

**DCGAN**

![Image Arithmetic](https://raw.githubusercontent.com/perticascatalin/HiddenDimensions/master/documentation/image_arithmetic.png "Image Arithmetic")

Radford et al. (2015)


### Samples Density and Principal Component Analysis

### Non-Linear Projections, Sub-Spaces and Clustering

### Navigation in Latent Spaces and Attractors



