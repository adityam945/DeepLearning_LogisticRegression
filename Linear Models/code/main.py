import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE

    # plt.scatter(X[:, 1:2], y, c="blue")

    plt.scatter(X[:, 0], y)
    plt.scatter(X[:, 1], y)

    plt.savefig("train_features.png")
    ### END YOUR CODE

def visualize_result(X, y, W):
	# '''This function is used to plot the sigmoid model after training. 

	# Args:
	# 	X: An array of shape [n_samples, 2].
	# 	y: An array of shape [n_samples,]. Only contains 1 or -1.
	# 	W: An array of shape [n_features,].
	
	# Returns:
	# 	No return. Save the plot to 'train_result_sigmoid.*' and include it
	# 	in submission.
	# '''
	### YOUR CODE HERE
    # y_pred = W
    # plt.scatter(X[:, 1], y[:, ], marker='.')
    # plt.plot(X[:, 1], y_pred, color='orange')
    # plt.show()
    # 
    clf = ((W[0] / W[2])) + (((W[0] / W[2])) * X)
    # plt.scatter(X[:, 0:1], y[:, ], marker='.', color='red')
    plt.plot(X[:, 1:3], y, clf, color='green')

    
    # 
    # plt.scatter(X[:, 1:3], y, W)
    # plt.scatter(X[:, 1], y)
    # 
    # plt.scatter(x=X[:, 1:3], y=y, s=W)
    # plt.show()

    plt.savefig("traninig-result.png")
	### END YOUR CODE

def visualize_result_multi(X, y, W):
	# '''This function is used to plot the softmax model after training. 

	# Args:
	# 	X: An array of shape [n_samples, 2].
	# 	y: An array of shape [n_samples,]. Only contains 0,1,2.
	# 	W: An array of shape [n_features, 3].
	
	# Returns:
	# 	No return. Save the plot to 'train_result_softmax.*' and include it
	# 	in submission.
	# '''
	### YOUR CODE HERE
    plt.scatter(X, y, c="red")
    plt.show()

	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 
#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------
    #arrayLR_classifier = []
    # arrayLR_Type = []
    prediction_object = {}    

   ##### Check BGD, SGD, miniBGD
    print("Default Parameters, learning_rate=0.5, max_iter=100")
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    print("Batch Gradient")
    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["ParameterDefault_fit_BGD"] = logisticR_classifier

    print("Mini-Batch Gradient")
    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["ParameterDefault_fit_miniBGD_1350_shape"] = logisticR_classifier

    print("Stochastic Gradident Descent")
    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["ParameterDefault_fit_SGD"] = logisticR_classifier

    print("Mini-Batch Gradient - 1")
    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["ParameterDefault_fit_miniBGD_1"] = logisticR_classifier

    print("Mini-Batch Gradient - 10")
    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["ParameterDefault_fit_miniBGD_10"] = logisticR_classifier


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    print("Parameter - 1, LR = 0.01 and max_iter = 100" )
    logisticR_classifier = logistic_regression(learning_rate=0.01, max_iter=100)

    print("Batch Gradient")
    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter1_fit_BGD"] = logisticR_classifier

    print("Mini-Batch Gradient")
    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter1_fit_miniBGD_1350_shape"] = logisticR_classifier

    print("Stochastic Gradident Descent")
    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter1_fit_SGD"] = logisticR_classifier

    print("Mini-Batch Gradient - 1")
    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter1_fit_miniBGD_1"] = logisticR_classifier

    print("Mini-Batch Gradient - 10")
    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter1_fit_miniBGD_10"] = logisticR_classifier
    # 
    print("Parameter - 2, LR = 0.1 and max_iter = 1000" )
    logisticR_classifier = logistic_regression(learning_rate=0.1, max_iter=1000)

    print("Batch Gradient")
    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter2_fit_BGD"] = logisticR_classifier

    print("Mini-Batch Gradient")
    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter2_fit_miniBGD_1350_shape"] = logisticR_classifier

    print("Stochastic Gradident Descent")
    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter2_fit_SGD"] = logisticR_classifier

    print("Mini-Batch Gradient - 1")
    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter2_fit_miniBGD_1"] = logisticR_classifier

    print("Mini-Batch Gradient - 10")
    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter2_fit_miniBGD_10"] = logisticR_classifier
    
    #
    print("Parameter - 3, LR = 0.5 and max_iter = 1000" )
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=1000)

    print("Batch Gradient")
    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter3_fit_BGD"] = logisticR_classifier

    print("Mini-Batch Gradient")
    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter3_fit_miniBGD_1350_shape"] = logisticR_classifier

    print("Stochastic Gradident Descent")
    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter3_fit_SGD"] = logisticR_classifier

    print("Mini-Batch Gradient - 1")
    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter3_fit_miniBGD_1"] = logisticR_classifier

    print("Mini-Batch Gradient - 10")
    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    prediction_object["Parameter3_fit_miniBGD_10"] = logisticR_classifier
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
    # visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    # visualize_result(train_X[:, 1:3], train_y, logisticR_classifier.score(train_X, train_y))

    ### YOUR CODE HERE
    # best training
    # print(prediction_object.keys())
    # print(prediction_object)
    # for key in prediction_object.items():
    #     prediction = prediction_object[key].score(train_X, train_y)
    # 
    best_model = 0
    best_model_key = ""
    # best_prediction = None

    for key, value in prediction_object.items():
        current_prediction = value.score(train_X, train_y)
        if current_prediction > best_model:
            best_model_key = key
            best_model = current_prediction
    
    # best model
    visualize_result(train_X[:,1:3], train_y, prediction_object[best_model_key].get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    print("The 'best' model above to do testing " + best_model_key)
    print(best_model)
    ### END YOUR CODE



    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
