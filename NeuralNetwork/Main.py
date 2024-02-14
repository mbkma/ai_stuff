import DataLoader
from Network import Network

training_data, validation_data, test_data = DataLoader.load_data_wrapper()
net = Network([784, 30, 10])
net.SGD(training_data, 10, 10, 3.0)
net.save("models/mnist.json")