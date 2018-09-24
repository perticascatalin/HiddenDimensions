from tensorflow.examples.tutorials import mnist
from pathlib import Path

curr_dir = Path(__file__).parent

mnist_data = mnist.input_data.read_data_sets(str(Path(curr_dir / 'data' / 'mnist')))

data = mnist.input_data.read_data_sets(str(Path(curr_dir / 'data' / 'fashion-mnist')), source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
