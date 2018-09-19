from tensorflow.examples.tutorials import mnist

mnist_data = mnist.input_data.read_data_sets(
    'C:/Users/Andrei Popovici/Documents/GitHub/HiddenDimensions/code/image_embedding/tensorflow-generative-model-collections-master/data/mnist')

data = mnist.input_data.read_data_sets('C:/Users/Andrei Popovici/Documents/GitHub/HiddenDimensions/code/image_embedding/tensorflow-generative-model-collections-master/data/fashion_mnist',
                                       source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
