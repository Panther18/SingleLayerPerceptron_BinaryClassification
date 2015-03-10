__author__ = 'Panther'


class DataLoader:

    """
    This function takes a file path as input and returns
        1. Number of attributes
        2. Names of each attribute
    It assumes that the first line of the file has the attributes information
    """
    @staticmethod
    def get_attributes(file_path, delimiter="\t"):
        try:
            with open(file_path, "r") as training_file:
                attribute_list = []
                temp_attribute_list = training_file.readline()  # Read the first line to count # of attributes
                temp_attribute_list = temp_attribute_list.split(delimiter)
                number_of_attributes = len(temp_attribute_list)
                if temp_attribute_list[len(temp_attribute_list) - 1 ] == '\n':  # Remove if the last element is '\n'
                    number_of_attributes -= 1
                each = 0
                while each != number_of_attributes:
                    attribute_list.append(temp_attribute_list[each])
                    each += 1
                training_file.close()
                return number_of_attributes, attribute_list
        except FileNotFoundError:
            print("Error in reading file")
            exit(0)

    """
    This function takes the following input
        1. File path as an input
        2. Number of attributes
        3. ignore_first_line = 1. This ignores the first lines which usually holds the attributes information
    And returns:
        1. List of lists where each outer list is training example and each inner list holds the training set values
    """
    @staticmethod
    def get_training_list(file_path, number_of_attributes, ignore_first_line=True, delimiter="\t"):
        class_index = number_of_attributes
        try:
            with open(file_path, "r") as training_file:
                if ignore_first_line:
                    training_file.readline()  # ignore the first line of the file
                training_list = []
                for each_record in training_file:
                    each_record = each_record.split(delimiter)
                    # Strip the \n from the last value
                    each_record[class_index] = each_record[class_index].split('\n')[0]
                    training_list.append(each_record)
                training_file.close()
                return training_list

        except FileNotFoundError:
            print('ERROR: Cannot open the file ' + training_file)
            exit(0)