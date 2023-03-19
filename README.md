# DNA_analyzing_network
The prokect is used to create and train neural networks, which predict some thermodynamic parameters of input dna secuence.
The project consists of several modules, modules implement different subtasks.
data_loader - used for load data, taken from the excel file and transform this data in features.
loss - contain several types of custom loss functions using for network training.
network_classes - contain several types of network arcitectures.
network_manager - contain functions for network training.
prediction_analyzer - contain functions and classes for prediction analysis, for example draw distribution of prediction error, or save prediction to some file.

model_training_example and model_testing_example shows examples of using modules for network training, saving, loading, and predinction analysis.
