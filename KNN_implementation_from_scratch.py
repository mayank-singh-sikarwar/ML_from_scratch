#CS6405_project_1: KNN implementation

#pre-requisite commands: below commands should be executed before running other part of the code
import numpy as np
import pandas as pd
import seaborn as sb 
import matplotlib.pyplot as plt

#####################################################################
def calculateDistance(dataSet, query_point):
    '''
    about: standard eclidian distance formula implementation
    input: training dataframe and query 
    output: array of euclidian distances between training set and query point
    '''
    dist = np.linalg.norm(query_point - dataSet, axis = 1) # square root of sum of squared difference 
                                                           # between two data points
    return dist
#####################################################################
def calculateManhattenDistances(train_feature, query_point):
    '''
    about: Manhatten distance formula implementation
    input: training dataframe and query 
    output: array of Manhatten distances between training set and query point
    '''
    abs_diff = np.abs(train_feature.sub(query_point, axis=0))# sumation of absolute values of substraction  
    Manhat_dist = np.sum(abs_diff, axis = 1).to_numpy() # between two data points 
    return Manhat_dist

#####################################################################
def getFeatureSet(data_frame):
    '''
    about: segregate features from data frame 
    input: data frame
    output: data frame of features
    '''
    return data_frame[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides","free sulfur dioxide","total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]

def getLabelSet(data_frame):
    '''
    about: segregate labels from data frame 
    input: data frame
    output: data frame of labels
    '''
    return data_frame["Quality"]
#####################################################################
def Weightedknn(k, dist_array, train_labels,prediction_weighted):
    '''
    about: prediction of class on the basis of Weighted k nearest neighbour implementation
    input: k value, distance array , data frame of training labels and empty array for stroing prediction
    output: array of predicted class of test data set
    '''
    temp = np.argpartition(dist_array, k)  # sorting the index of min. distances
    kmin_index = temp[:k] #indices of k min values 
    label = np.unique(train_labels[kmin_index].to_numpy()) #collecting labels
    #initializing weights
    w1 = 0 
    w2 = 0
    
    for j in kmin_index: #iterating through indices of min. distances
        if train_labels[j] == label[0]: #segregating weights on label basis
            d = dist_array[j]
            if d != 0:
                w1 += 1/d**2
            else:
                w1 +=1
                
        else:
            d = dist_array[j]
            if d != 0:
                w2 += 1/d**2
            else:
                w2 +=1
        
    if w1 > w2: #weight comparison
        predict_class = label[0] #assigning label
    else:
        predict_class = label[1]

    prediction_weighted = np.append(prediction_weighted, predict_class) #adding prediction result
    
    return prediction_weighted
#####################################################################
def znormalization(train_feature, test_feature):
    '''
    about: z- normalization implementation
    input: training data frame and test data frame
    output: normalized training data frame and test data frame
    '''
    #normalizing training data set
    normtrain_feature = (train_feature - train_feature.mean(axis = 0))/(np.std(train_feature.to_numpy(), axis = 0 , dtype = np.float64))
    #normalizing test data set
    normtest_feature = (test_feature - train_feature.mean(axis = 0))/(np.std(train_feature.to_numpy(), axis = 0 , dtype = np.float64))
        
    return normtrain_feature, normtest_feature
##################################################################################################
def minmaxnormalization(train_feature, test_feature):
    '''
    about: min. max. value normalization implementation
    input: training data frame and test data frame
    output: normalized training data frame and test data frame
    '''
    #normalizing training data set
    normtrain_feature = (train_feature - train_feature.min(axis = 0))/(train_feature.max(axis = 0) - train_feature.min(axis = 0))
    #normalizing test data set
    normtest_feature = (test_feature - train_feature.min(axis = 0))/(train_feature.max(axis = 0) - train_feature.min(axis = 0))  
        
    return normtrain_feature, normtest_feature
###################################################################################################
def knn(k,normalize = None, distance = 'Euclid'):
    '''
    about: read data from the current workinmg directory and performs accuracy calculation for knn and W.knn method
    input: number of neighbours to check, normalization method(default: non normalized data), calculation method 
    for distance of the nearest neighbours(default: Euclidian distance).
    output: accuracy of knn & and accuracy of W.knn
    '''
    #reading training and test data and segragating features and label for training
    #and test data set
    df_train = pd.read_csv("dataset-p1/wine-data-project-train.csv")
    df_test  = pd.read_csv("dataset-p1/wine-data-project-test.csv")
    train_feature = getFeatureSet(df_train)
    test_feature = getFeatureSet(df_test)
    train_labels = getLabelSet(df_train) 
    test_labels = getLabelSet(df_test)
    #initializing variables 
    prediction = np.array([])  
    prediction_weighted = np.array([])
    total_correct = 0
    total_correct_weighted = 0
    accuracy = 0
    accuracyWeightedModel = 0
    
    #check for normalization
    if normalize is not None:
        if normalize == 'znorm':
            train_feature, test_feature = znormalization(train_feature, test_feature)
        elif normalize == 'maxmin':
            train_feature, test_feature = minmaxnormalization(train_feature, test_feature)
        else:
            pass
    
    #iterating through test data for euclidian distance calculation
    for i in range(0,len(test_feature)):
        
        #signle query point taken
        query_point = test_feature.iloc[[i]].to_numpy() 
        query_point =pd.DataFrame(np.repeat(query_point,len(train_feature), axis=0), columns= test_feature.columns)
        
        #temporary empty array for holding distance
        dist_array = np.array([])
        #check for method of calculating distance
        if distance == 'Euclid':
            #calculation distance along row
            dist = calculateDistance(train_feature, query_point)
            dist_array = np.append(dist_array, dist) 
        else:
            #calculation distance along row
            dist = calculateManhattenDistances(train_feature, query_point)
            dist_array = np.append(dist_array, dist) 
        
        # sorting the index of min. distances   
        temp = np.argpartition(dist_array, k)  
        kmin_index = temp[:k] 
        
        #counting labels
        label, count = np.unique(train_labels[kmin_index].to_numpy(), return_counts = True) 
        
        #majority voting
        predict_class = label[np.argmax(count)] 
        prediction = np.append(prediction, predict_class)
        
        #calling weightedknn function
        prediction_weighted = Weightedknn(k,dist_array,train_labels,prediction_weighted) 
        
        #prediction validation for knn method
        if prediction[i] == test_labels[i]: 
            total_correct += 1
        
        #prediction validation for weighted knn method    
        if prediction_weighted[i] == test_labels[i]:
            total_correct_weighted += 1
    
    #accuracy calculation for knn        
    accuracy = (total_correct/len(test_labels))*100 
    
    #accuracy calculation for Wknn
    accuracyWeightedModel = (total_correct_weighted/len(test_labels))*100 
    
    return accuracy, accuracyWeightedModel
###################################################################################################
def main(lower_k, higher_k, normalize, distance):
    '''
    about: Calculates the accuracy at different k values and returns 
    input: lower and higher values of nearest neighbour , normalization method(default: non normalized data), 
    calculation method for distance of the nearest neighbours.
    output: accuracy of knn & and accuracy of W.knn
    '''
    
    #intiating empty dictionaries and list of k values
    allResults = {} 
    allResultsweighted = {}
    k_values = list(range(lower_k, higher_k,2))
    
    #iterating through k values
    for k in k_values:
        #calling knn function
        accuracy, accuracyWeightedModel = knn(k+1, normalize, distance) 
        allResults[k+1] = accuracy 
        allResultsweighted[k+1] = accuracyWeightedModel 
    
    #list of keys   
    y_keys = list(allResults.keys())  
    y2_keys = list(allResultsweighted.keys())
    
    #accuracy of knn & W.knn at different k
    x_values = list(allResults.values()) 
    x2_values = list(allResultsweighted.values()) 
    
    #max. accuracy obtained through knn and W.knn
    max_val = max(x_values)
    max_val_w = max(x2_values) 
    
    #key of max. accuracy obtained through knn
    max_key = max(allResults, key=allResults.get) 
    max_key_w = max(allResultsweighted, key=allResultsweighted.get) 
    
    #output
    print(f'For knn method, maximum accuracy at k ={max_key} and accuracy is {max_val:.2f}% .')
    print(f'For weight knn method, maximum accuracy at k ={max_key_w} and accuracy is {max_val_w:.2f}% .')
    sb.set_style("darkgrid")
    plt.figure(figsize = (9, 4))
    plt.plot( y_keys, x_values, color = 'red', label = "Knn Accuracy")
    plt.plot(y2_keys, x2_values,label = "Weighted Knn Accuracy")
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy performance on multiple k')
    plt.legend()
    plt.show()
###################################################################################################
#function calling
main(2,40,None,"Euclid")
main(2,40,None,"Manhatten")
main(2,40,'znorm',"Euclid")
main(2,40,'znorm',"Manhatten")
main(2,40,'maxmin',"Euclid")
main(2,40,'maxmin',"Manhatten")