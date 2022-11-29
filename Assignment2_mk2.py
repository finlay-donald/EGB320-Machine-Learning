#Importing all the libraries
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2

import matplotlib.pyplot as plt
from skimage.io import imread_collection
from skimage.transform import resize
from keras.applications.mobilenet_v2 import preprocess_input


def image_input_process(image_folder):
    """
    Preprocesses all of the images in a given subfolder of the small_flower_dataset
    
    @param
        A string of the name of the subfolder containing all the images to preprocess

    @return
       An array of the preprocessed images, type: float64
    """
    
    image_collection = imread_collection("./small_flower_dataset/"+ image_folder + "/*.jpg")
    out_data = np.empty((len(image_collection), 224, 224, 3))
    for i in range(len(image_collection)):
        temp = preprocess_input(image_collection[i])
        out_data[i] = resize(temp, output_shape=(224, 224))
        
    return out_data

def make_total_dataset(*data_sets):
    """
    Generate a single sataset from multiple input datasets, as well as generate corresponding labels
    
    @param
        Arrays containing the images

    @return
       The combined dataset, and the combined datasets corresponding labels
    """
    #First getting the total number of images contained within all the data sets
    num_data = 0
    for dataset in data_sets:
        num_data += len(dataset)
    
    #Generate an empty array for our data, MobileNet V2 needs images of size 224, 224, 3
    data = np.empty((num_data, 224, 224, 3))
    #Generate an empty array for the labels of our dataset
    labels = np.empty(num_data, dtype=int)
    
    count = 0
    prev_dataset_num = 0
    for dataset in data_sets:
        #Assigning the current dataset in the loop to the corresponding indices of the data array
        data[prev_dataset_num:(prev_dataset_num + len(dataset))] = dataset[0:]\
        #Assigning the current dataset label to the corresponding labels indices
        #Starts at a label of 0, and incriments in 1, all the way to the number of datasets provided-1
        labels[prev_dataset_num:(prev_dataset_num + len(dataset))] = count
        prev_dataset_num += len(dataset)
        count += 1
    
    return data, labels

def split_data(data, labels, train_percent, val_percent, test_percent):
    """
    Split the data into training, validation, and testing datasets
    
    @param
        The data to be split, the corresponding labels for that data, the percent of data to be used for training, the percent of data to be used for validation, and the percent of data to be used for testing

    @return
       Arrays of training, validation, and testing data, as well as training, validation, and testing labels
    """
    from sklearn.model_selection import train_test_split
    #Checking that all of the percentages add to one
    if train_percent + val_percent + test_percent != 100:
        raise NameError("Error, the percentages must sum to 1")
    #Splitting the data into a training dataset, and the remainder for the validation and testing
    data_train, data_val_and_test, labels_train, labels_val_and_test = train_test_split(data, labels, test_size = (100 - train_percent)/100)
    #Splitting the remainder into validation and test datasets
    data_val, data_test, labels_val, labels_test = train_test_split(data_val_and_test, labels_val_and_test, test_size = (test_percent/(val_percent + test_percent)))
    
    return data_train, data_val, data_test, labels_train, labels_val, labels_test
    

#Reading all the datasets to varaibles, and preprocessing them
daisies = image_input_process("daisy")
dandelions = image_input_process("dandelion")
roses = image_input_process("roses")
sunflowers = image_input_process("sunflowers")
tulips = image_input_process("tulips")

#Creating a single output dataset, with corresponding labels
data, labels = make_total_dataset(daisies, dandelions, roses, sunflowers, tulips)


from keras import Model
from keras.layers import Dense


# Testing 
test_LR = True # Set to true to test multiple learning rates
test_M = False # Set to true to test multiple Momentum values

# initialize data variables to save data for each test
data_accuracy = [None] * 5
data_val_accuracy = [None] * 5
data_loss = [None] * 5
data_val_loss = [None] * 5
fine_tune_data_accuracy = [None] * 5
fine_tune_data_val_accuracy = [None] * 5
fine_tune_data_loss = [None] * 5
fine_tune_data_val_loss = [None] * 5

#Initialising the learning rate and momentum - Should be changed below
LR = [0.05]
M = [0.8]

#Generating a vector for the learning rates, and accompanying momentums (all zero)
#Set to True if testing the learning rate
if test_LR == True:
    LR = [0.01, 0.05, 0.1, 0.2, 0.5] # Vary the learning rate
    M = [0.0, 0.0, 0.0, 0.0, 0.0] # make M a vector so that indexing works

#Generating a vector for the learning rates (Using the best learning rate of 0.2), and accompanying momentums
#Set to True if testing the momentum
if test_M == True:
    M = [0.0, 0.1, 0.4, 0.8, 1]
    LR = [0.2, 0.2, 0.2, 0.2, 0.2] # make LR a vector so that indexing works

#Initialising a counter for the loop
i = 0
#Using a for loop to iterate through each of the changed learning rate or momentum
for L in LR:
    if test_M and test_LR == True:
        error = ', Change either test_LR or test_M to False'
        raise ValueError('Please test learning rate and momentum seperately'+error)
    
    #Generate a base model using MobileNet V2
    model = MobileNetV2(weights = "imagenet")
    
    
    """
    Creating a prediction for our dataset with the current trained model (not training using our data set)
    """
    from keras.applications.mobilenet_v2 import decode_predictions

    untrained_prediction = False
    if untrained_prediction:
        predictions = model.predict(data)
        for decode_prediction in decode_predictions(predictions, top=1):
            for name, desc, score in decode_prediction:
                print("- {} ({:.2f}%)".format(desc, 100*score))
    
    """
    Modifying the model with our dataset
    """
    flower_output = Dense(5, activation="softmax")
    flower_output = flower_output(model.layers[-2].output)

    flower_input = model.input
    flower_model = Model(inputs = flower_input, outputs = flower_output)

    #Freezing the layers which are not to be trained
    for layer in flower_model.layers[:-1]:
        layer.trainable = False
    
    
    """
    Compiling the model
    """
    flower_model.compile(
        loss = "sparse_categorical_crossentropy",
        optimizer = tf.keras.optimizers.SGD(learning_rate = L, momentum = M[i], nesterov = False),
        metrics = ["accuracy"]
    )
    
    
    """
    Generating a callback function to stop the fitting
    """
    callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", mode="min", patience = 5, verbose = 1, restore_best_weights = True)
    
    
    unvalidated = False
    if unvalidated:
        """
        Training the model, without a validation dataset
        """
        history_unvalidated = flower_model.fit(
            x = data,
            y = labels,
            epochs = 20,
            verbose = 2
            )
        
        
        """
        Creating the predictions based on the un-validated data
        """
        predictions = flower_model.predict(data)
        
        np.argmax(predictions, axis=1)
        
    
    """
    Generating seperate training, validation, and testing datasets
    """
    training_data, validation_data, test_data, training_labels, validation_labels, test_labels = split_data(data, labels, 70, 15, 15)
    
    validate = True
    if validate:
        history_validated = flower_model.fit(
            x = training_data,
            y = training_labels,
            epochs = 30,
            verbose = 1,
            validation_data = (validation_data, validation_labels),
            callbacks = [callback]
        )
    
    
    """
    Fine tuning the model
    """
    fine_tuning = True
    if fine_tuning:
        flower_model.trainable = True
        #Try different algorithm for fine tuning such as adam, set initial low learning rate
        flower_model.compile(
            loss = "sparse_categorical_crossentropy",
            optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5),
            metrics = ["accuracy"]
        )
        
        history_fine_tune = flower_model.fit(
            x = training_data,
            y = training_labels,
            epochs = 10,
            verbose = 1,
            validation_data = (validation_data, validation_labels),
            callbacks = [callback]
        )
    
    
    """
    Testing the model using our test dataset
    """
    trained_prediction = True
    if trained_prediction:
        predictions = flower_model.predict(test_data)
        truths = np.argmax(predictions, axis=1) == test_labels
        print(truths)
        print("The model correctly predicted {}% of the test dataset".format((1-(np.size(truths) - np.count_nonzero(truths))/(np.size(truths)))*100))
    
    
    """
    Testing the model MK2
    """
    score = flower_model.evaluate(test_data, test_labels, verbose = 1)


    """
    Save all test data
    """
    data_accuracy[i] = history_validated.history['accuracy']
    data_val_accuracy[i] = history_validated.history['val_accuracy']
    data_loss[i] = history_validated.history['loss']
    data_val_loss[i] = history_validated.history['val_loss']
    fine_tune_data_accuracy[i] = history_fine_tune.history['accuracy']
    fine_tune_data_val_accuracy[i] = history_fine_tune.history['val_accuracy']
    fine_tune_data_loss[i] = history_fine_tune.history['loss']
    fine_tune_data_val_loss[i] = history_fine_tune.history['val_loss']
    
    i += 1 # iterate

# Prepare variables for figures
if test_LR == True:
    test1 = 'Learning Rate: ' + str(LR[0])
    test2 = 'Learning Rate: ' + str(LR[1])
    test3 = 'Learning Rate: ' + str(LR[2])
    test4 = 'Learning Rate: ' + str(LR[3])
    test5 = 'Learning Rate: ' + str(LR[4])

if test_M == True:
    test1 = 'Momentum: ' + str(M[0])
    test2 = 'Momentum: ' + str(M[1])
    test3 = 'Momentum: ' + str(M[2])
    test4 = 'Momentum: ' + str(M[3])
    test5 = 'Momentum: ' + str(M[4])
    

if test_LR or test_M: # Only graph information this way if a test was run
    for data in data_accuracy:
        plt.plot(data)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend([test1, test2, test3, test4, test5], bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show()
    
    for data in data_val_accuracy:
        plt.plot(data)
    plt.title('model validation accuracy')
    plt.ylabel('validation accuracy')
    plt.xlabel('epoch')
    plt.legend([test1, test2, test3, test4, test5], bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show()
    
    for data in data_loss:
        plt.plot(data)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend([test1, test2, test3, test4, test5], bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show()
    
    for data in data_val_loss:
        plt.plot(data)
    plt.title('model validation loss')
    plt.ylabel('validation loss')
    plt.xlabel('epoch')
    plt.legend([test1, test2, test3, test4, test5], bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show()
    
    for data in fine_tune_data_accuracy:
        plt.plot(data)
    plt.title('model fine tuning accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend([test1, test2, test3, test4, test5], bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show() 
    
    for data in fine_tune_data_val_accuracy:
        plt.plot(data)
    plt.title('model fine tuning validation accuracy')
    plt.ylabel('validation accuracy')
    plt.xlabel('epoch')
    plt.legend([test1, test2, test3, test4, test5], bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show() 
    
    for data in fine_tune_data_loss:
        plt.plot(data)
    plt.title('model fine tuning loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend([test1, test2, test3, test4, test5], bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show() 
    
    for data in fine_tune_data_val_loss:
        plt.plot(data)
    plt.title('model fine tuning validation loss')
    plt.ylabel('validation loss')
    plt.xlabel('epoch')
    plt.legend([test1, test2, test3, test4, test5], bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show() 
    
else: # if no tests were run plot values together
    if validate:
        plt.plot(history_validated.history['accuracy'])
        plt.plot(history_validated.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'],  bbox_to_anchor=(1.04,0.5), loc="center left")
        plt.show()
        
        plt.plot(history_validated.history['loss'])
        plt.plot(history_validated.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'],  bbox_to_anchor=(1.04,0.5), loc="center left")
        plt.show()
        
    
    
    if fine_tuning:
        plt.plot(history_fine_tune.history['accuracy'])
        plt.plot(history_fine_tune.history['val_accuracy'])
        plt.title('model fine tuning accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'],  bbox_to_anchor=(1.04,0.5), loc="center left")
        plt.show()
        
        plt.plot(history_fine_tune.history['loss'])
        plt.plot(history_fine_tune.history['val_loss'])
        plt.title('model fine tuning loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'],  bbox_to_anchor=(1.04,0.5), loc="center left")
        plt.show()









