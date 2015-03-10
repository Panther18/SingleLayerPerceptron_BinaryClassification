__author__ = 'Panther'
import Load_File
import GradientDescent
import sys
"""
This is a driver program

Read the training examples and store them
Assign the initial weights to zero
Repeat till termination:
    For each training examples, find the output with the current weights. Call it 'o'
    Find the sigmoid value 'o' using the function 1/(1+e^-o)
    Update the weight of each attribute using the equation w= w + (LR)(ERROR)(obj)(1-obj)xi where obj is the sigmoid fn
    and xj is the input value
"""
if __name__ == "__main__":
    if len(sys.argv) > 0:
        training_data_path = sys.argv[1]  # Read the four arguments
        test_data_path = sys.argv[2]
        learning_rate = float(sys.argv[3])
        iterations = int(sys.argv[4])
        number_of_attributes, attribute_list = Load_File.DataLoader.get_attributes(training_data_path)
        training_list = Load_File.DataLoader.get_training_list(training_data_path, number_of_attributes, delimiter="\t")
        gd = GradientDescent.GradientDescent(training_list, number_of_attributes, attribute_list, learning_rate)
        gd.find_weights(iterations)
        print('Accuracy (' + str(len(training_list)) + ' instances):'+ str(round(gd.find_accuracy(training_list), 2)))
        test_list = Load_File.DataLoader.get_training_list(test_data_path, number_of_attributes, delimiter="\t")
        print('Accuracy (' + str(len(test_list)) + ' instances):' + str(round(gd.find_accuracy(test_list), 2)))
    else:
        print("No training data to learn")