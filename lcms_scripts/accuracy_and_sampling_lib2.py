import numpy as np
import pandas as pd
import pdb
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.utils import parallel_backend
from sklearn.model_selection import learning_curve

#--------------------------------------------------------------
#               Accuracy Assessment Functions
#---------------------------------------------------------------
# These functions are based off Stephen V. Stehman (2014) Estimating area and map accuracy for stratified
# random sampling when the strata are different from the map classes, International Journal of
# Remote Sensing, 35:13, 4923-4939, DOI: 10.1080/01431161.2014.930207'

# That paper has an example dataset and explicit equations that were used to 
# develop these functions. Each of the core functions states which equation in the paper they were based off of. 
# It can be difficult to understand unless you sit down with the paper and puzzle through all the equations step-by-step.
# I encourage you to do this to double check what I have done!

#-----------------------Yu and Xu Equations-------------------------------------
# Yu for Overall Accuracy, Stehman et al. Equation 12
# Yu = 1 if pixel u is classified correctly
def overallAccuracyYu(map_class, reference_class):
    out_yu = []
    for i in range(len(map_class)):
        if map_class[i] == reference_class[i]:
            out_yu.append(1)
        else:
            out_yu.append(0)
    return out_yu

# Yu for Proportion of Area, Stehman et al. Equation 14
# Yu = 1 if pixel u is reference class k
def areaYu(reference_class, assessment_class):
    out_yu = []
    for i in range(len(reference_class)):
        if reference_class[i] == assessment_class:
            out_yu.append(1)
        else:
            out_yu.append(0)
    return out_yu

# Yu for Users Accuracy, Stehman et al. Equation 18
# Yu = 1 if pixel u is classified correctly and has map class k
def usersAccuracyYu(map_class, reference_class, assessment_class):
    out_yu = []
    for i in range(len(map_class)):
        if map_class[i] == reference_class[i] and map_class[i] == assessment_class:
            out_yu.append(1)
        else:
            out_yu.append(0)
    return out_yu

# Xu for Users Accuracy, Stehman et al. Equation 19 
# Xu = 1 if pixel u is map class k   
def usersAccuracyXu(map_class, assessment_class):
    out_xu = []
    for i in range(len(map_class)):
        if map_class[i] == assessment_class:
            out_xu.append(1)
        else:
            out_xu.append(0)
    return out_xu

# Yu for Producers Accuracy, Stehman et al. Equation 22
# Yu = 1 if pixel u is correctly classified and has reference class k
def producersAccuracyYu(map_class, reference_class, assessment_class):
    out_yu = []
    for i in range(len(map_class)):
        if map_class[i] == reference_class[i] and reference_class[i] == assessment_class:
            out_yu.append(1)
        else:
            out_yu.append(0)
    return out_yu

# Xu for Producers Accuracy, Stehman et al. Equation 23 
# Xu = 1 if pixel u is has reference class k
def producersAccuracyXu(reference_class, assessment_class):
    out_xu = []
    for i in range(len(reference_class)):
        if reference_class[i] == assessment_class:
            out_xu.append(1)
        else:
            out_xu.append(0)
    return out_xu


#-----------------------Fundamental Equations-------------------------------------

# Stehman et al. Equation 3
# Y = Population Mean
# strataDict must be a dictionary with the strata values as keys 
# and the total number of pixels in each stratum as values (N_h*)
# strataYuFrame is a dataframe with the strata values in one column and the yu values in the second
# OR it has 'reference_class' and 'map_class' columns instead of yu for balanced accuracy and kappa
def population_mean(strataDict, strataYuFrame, metric, multiclass = False):
    # print(metric)
    
    numerator_values = []
    denominator_values = []
    for h in strataDict.keys():
        thisStrat = strataYuFrame[strataYuFrame['strata'] == int(h)]
        N_h_star = np.float(strataDict[h])
        # if isinstance(N_h_star, int):
        #     N_h_star = np.int64(N_h_star)
        n_h_star = len(thisStrat)

        if n_h_star > 0:
            if metric == 'balanced_accuracy':
                sample_mean = metrics.balanced_accuracy_score(thisStrat['reference_class'], thisStrat['map_class'])
            elif metric == 'kappa':
                sample_mean = metrics.cohen_kappa_score(thisStrat['reference_class'], thisStrat['map_class']) # yh bar
            elif metric == 'f1_score':

                if multiclass: 
                    # Multi-Class
                    sample_mean = metrics.f1_score(thisStrat['reference_class'], thisStrat['map_class'], average='micro')
                else:
                    # Two-Class
                    sample_mean = metrics.f1_score(thisStrat['reference_class'], thisStrat['map_class'], average='binary', zero_division = 1)

            else:
                sample_mean = thisStrat['yu'].sum() / n_h_star # yh bar
            # print('Stratum '+str(h)+':', sample_mean)
            #print('n_h_star', n_h_star) 

            numerator_values.append(N_h_star * sample_mean)
            denominator_values.append(N_h_star)
        elif metric == 'Area' or metric == 'area':
            sample_mean = np.nan
            numerator_values.append(0)
            denominator_values.append(N_h_star)
        else:
            numerator_values.append(N_h_star)
            denominator_values.append(N_h_star)
    weights = []
    for i in range(0, len(numerator_values)):
        # print('Stratum '+str(i+1)+' Weight:', numerator_values[i]/np.sum(denominator_values))
        weights.append(numerator_values[i]/np.sum(denominator_values))
    # print('Numerator Values: ', numerator_values)
    # print('Weights: ', weights)
    # print('Weights sum:', np.sum(weights))
    out = np.sum(numerator_values) / np.sum(denominator_values)
    # print('Out sum', out)
    return out

# Stehman et al. Equation 25
# Variance of Y
def variance(strataDict, strataYuFrame):
    numerator_values = []
    denominator_values = []
    for h in strataDict.keys():
        thisStrat = strataYuFrame[strataYuFrame['strata'] == int(h)].reset_index()
        N_h_star = np.float(strataDict[h])
        # if isinstance(N_h_star, int):
        #     N_h_star = np.int64(N_h_star)
        n_h_star = len(thisStrat)
        
        if n_h_star > 0:
            sample_mean = thisStrat['yu'].sum() / n_h_star # yh bar

            # Sample Variance = Stehman et al. Equation 26
            thisStrat['sampleVar'] = (thisStrat['yu'].subtract(sample_mean)).pow(2).divide(n_h_star - 1)
            sampleVariance = thisStrat['sampleVar'].sum()

            numerator_values.append(N_h_star**2 * (1 - n_h_star/N_h_star) * sampleVariance / n_h_star)
            if N_h_star**2 * (1 - n_h_star/N_h_star) * sampleVariance / n_h_star < 0:
                # If this value is below zero, there is likely an issue with the type of your numbers.
                pdb.set_trace()
            denominator_values.append(N_h_star)

        else:
            sample_mean = np.nan
            numerator_values.append(0)
            denominator_values.append(N_h_star)
    
    # N in equation 25 = the sum of all the strata values, so this is the same as np.sum(denominator_values)
    out = (1 / (np.sum(denominator_values)**2)) * np.sum(numerator_values)
    return out

# Standard Error
def standard_error(variance):
    return np.sqrt(variance)

# Stehman et al. Equation 20
# R = Ratio for Users or Producers Accuracy
# strataDict must be a dictionary with the strata values as keys 
# and the total number of pixels in each stratum as values (N_h*)
# strataYuFrame is a dataframe with columns ['strata','yu','xu']
def users_producers_ratio(strataDict, strataYuFrame, metric):
    # print(metric)
    numerator_values = []
    denominator_values = []
    for h in strataDict.keys():
        thisStrat = strataYuFrame[strataYuFrame['strata'] == int(h)]
        N_h_star = np.float(strataDict[h])
        # if isinstance(N_h_star, int):
        #     N_h_star = np.int64(N_h_star)
        n_h_star = len(thisStrat)

        sample_mean_yu = thisStrat['yu'].sum() / n_h_star # yh bar
        sample_mean_xu = thisStrat['xu'].sum() / n_h_star # xh bar

        # print('Stratum '+str(h)+': R=', sample_mean_yu/sample_mean_xu)

        if not np.isnan(sample_mean_xu) and not np.isnan(sample_mean_yu):
            numerator_values.append(N_h_star * sample_mean_yu)
            denominator_values.append(N_h_star * sample_mean_xu)

    out = np.sum(numerator_values) / np.sum(denominator_values)
    return out

#---------------------------------Metric Wrappers------------------------------------
# Overall Accuracy
def get_overall_accuracy(reference_class, map_class, strata_class, strataDict):

    strataYuFrame = pd.DataFrame({'strata': strata_class,
                                    'yu': overallAccuracyYu(map_class, reference_class)})
    
    overall_accuracy = population_mean(strataDict, strataYuFrame, 'overall_accuracy')
    standard_error = np.sqrt(variance(strataDict, strataYuFrame)) 
    
    # print('Overall Accuracy: ', overall_accuracy, '+/-', standard_error)

    return overall_accuracy, standard_error

# Kappa, Balanced Accuracy, F1 Score
def get_other_accuracy_metric(reference_class, map_class, strata_class, strataDict, metric, multiClass = False):

    strataYuFrame = pd.DataFrame({'strata': strata_class,
                            'map_class': map_class,
                            'reference_class': reference_class})
    
    overall_metric = population_mean(strataDict, strataYuFrame, metric, multiClass)
    
    # print('Overall '+metric+': ', overall_metric)

    return overall_metric

# Users and Producers Accuracy
def get_users_producers_accuracy(reference_class, map_class, strata_class, strataDict, assessment_classes):
    usersOut = {}
    producersOut = {}
    usersError = {}
    producersError = {}
    for assessment_class in assessment_classes:
        # print('Class: ', assessment_class)
        # Users Accuracy
        usersYuXuFrame = pd.DataFrame({'strata': strata_class, 
                            'yu': usersAccuracyYu(map_class, reference_class, assessment_class),
                            'xu': usersAccuracyXu(map_class, assessment_class)})

        # Producers Accuracy
        producersYuXuFrame = pd.DataFrame({'strata': strata_class,
                            'yu': producersAccuracyYu(map_class, reference_class, assessment_class),
                            'xu': producersAccuracyXu(reference_class, assessment_class)})
        
        
        usersOut[assessment_class] = users_producers_ratio(strataDict, usersYuXuFrame, 'Users Accuracy')
        producersOut[assessment_class] = users_producers_ratio(strataDict, producersYuXuFrame, 'Producers Accuracy')
        usersError[assessment_class] = np.sqrt(variance(strataDict, usersYuXuFrame))
        producersError[assessment_class] = np.sqrt(variance(strataDict, producersYuXuFrame))
        
        # print('Users Accuracy for Class '+str(assessment_class)+': ', usersOut[assessment_class], '+/-', standard_error)
        # print('Producers Accuracy for Class '+str(assessment_class)+': ', producersOut[assessment_class], '+/-', standard_error)
        # print('')
    return usersOut, producersOut, usersError, producersError

# Area Estimation
def get_area_estimation(reference_class, strata_class, strataDict, assessment_classes):

    areas = {}
    errors = {}
    for assessment_class in assessment_classes:
        try:
            strataYuFrame = pd.DataFrame({'strata': strata_class,'yu': areaYu(reference_class, assessment_class)})
        except:
            pdb.set_trace()

        areas[assessment_class] = population_mean(strataDict, strataYuFrame, 'Area')
        classVariance = variance(strataDict, strataYuFrame)
        errors[assessment_class] = np.sqrt(classVariance)

        # print('Estimated Area for Class '+str(assessment_class)+': ', areas[assessment_class])
        if np.isnan(np.sqrt(classVariance)):
            print('Check Variance')
            pdb.set_trace()

    return areas, errors

#---------------------------Plotting Functions-----------------------------------------
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, model):
    fig = plt.figure()
    plt.title(model+' Threshold vs. Precision/Recall')
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.ylim([0,1])
    return plt, fig

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), scoring = 'balanced_accuracy'):

    fig = plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Samples")
    plt.ylabel("Score")

    with parallel_backend('threading'):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring, shuffle = True)
    # print('train sizes: ')
    # print(train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt, fig



#---------------------------------Write to File Wrappers------------------------
# Wrapper script for calculating all the accuracy metrics in one place, and optionally writing it to file
def get_write_stratified_accuracies(\
    y_train,            # The correct classifications
    y_predictions,      # The predicted classifications
    strata_train,       # The strata of the same plots as above
    strataDict,         # Dictionary of the number of pixels in each stratum - defined in LCMSVariables - used for weighting
    assessment_classes, # Class names - used for looping through classes for users/producers accuracies and areas
    assessment_class_names,
    method = '',        # This is just a run name, used for printing out accuracies in file. Not really used anymore
    accFile = None):    # Path of txt file you want to write the accuracies directly to file here.

    # Is this a multi-class or binary model?
    if y_train.nunique() > 2:
        multiClass = True
    else:
        multiClass = False

    # Get accuracy metrics. These functions all follow Stehman 2014. That paper has an example dataset and explicit equations that were used to 
    # develop these functions. Each of the core functions above states which equation in the paper they were based off of. 
    # It can be difficult to understand unless you sit down with the paper and puzzle through all the equations step-by-step.
    # I encourage you to do this to double check what I have done!
    accuracy, accuracy_error = get_overall_accuracy(y_train.to_numpy(), y_predictions, strata_train, strataDict)  
    balanced_accuracy = get_other_accuracy_metric(y_train.to_numpy(), y_predictions, strata_train, strataDict,'balanced_accuracy') 
    kappa = get_other_accuracy_metric(y_train.to_numpy(), y_predictions, strata_train, strataDict,'kappa')     
    f1_score = get_other_accuracy_metric(y_train.to_numpy(), y_predictions, strata_train, strataDict, 'f1_score', multiClass)                              
    users, producers, usersError, producersError = get_users_producers_accuracy(y_train.to_numpy(), y_predictions, strata_train, strataDict, assessment_classes)
    areas, area_errors = get_area_estimation(y_train.to_numpy(), strata_train, strataDict, assessment_classes)                   

    class_dict = dict(list(zip(assessment_classes,assessment_class_names)))
    # Print accuracy results
    # print(method+' - Accuracy: ', accuracy, '+/-', accuracy_error)
    # print(method+' - Balanced Accuracy: ', balanced_accuracy)
    # print(method+' - Kappa: ', kappa)
    # print(method+' - F1 Score: ', f1_score)
    # print(method+' - Users Accuracy: ')
    # for c in users.keys():
    #     print('Class '+str(c)+': '+str(users[c]), '+/-', usersError[c])
    # print(method+' - Producers Accuracy: ') 
    # for c in producers.keys():
    #     print('Class '+str(c)+': '+str(producers[c]), '+/-', producersError[c])                   
    # print()
    # print(method+' - Design-Based Area Estimation: ') 
    # for c in areas.keys():
    #     print('Class '+str(c)+': '+str(areas[c]), '+/-', area_errors[c])                   
    # print()

    # Write to File
    if accFile != None:
        accFile.write('Accuracy: '+str(accuracy)+' +/- '+str(accuracy_error)+'\n')
        accFile.write('Balanced Accuracy: '+str(balanced_accuracy)+'\n')
        accFile.write('Kappa: '+str(kappa)+'\n')
        accFile.write('F1 Score: '+str(f1_score)+'\n')
        accFile.write('Users Accuracy: \n')
        for c in users.keys():
            accFile.write('Class '+str(class_dict[c])+': '+str(users[c])+' +/- '+str(usersError[c])+'\n')
        accFile.write('Producers Accuracy: \n')
        for c in producers.keys():
            accFile.write('Class '+str(class_dict[c])+': '+str(producers[c])+' +/- '+str(producersError[c])+'\n')
        accFile.write('Design-Based Area Estimation: \n')
        for c in areas.keys():
            accFile.write('Class '+str(class_dict[c])+': '+str(areas[c])+' +/- '+str(area_errors[c])+'\n')
    
    
    return accuracy, balanced_accuracy, users, producers, kappa, f1_score, areas, accuracy_error, usersError, producersError, area_errors
###################################################
# Function to get a weighted confusion matrix
def getConfusionMatrix(ref,pred,strata_class, stratum_weights, strataDict,assessment_classes,labels,nan_value = -99):
    # Set up labels
    row_labels = labels.copy()
    row_labels.append("Producer's Accuracy")

    col_labels = labels.copy()
    col_labels.append("User's Accuracy")

    # Get raw weighted confusion matrix
    cm = metrics.confusion_matrix(ref,pred,sample_weight=stratum_weights).T
   
    # Get accuracy metrics using Stehman 2014 methods
    overall_accuracy, accuracy_error = get_overall_accuracy(ref.to_numpy(), pred, strata_class, strataDict)  

    users, producers, usersError, producersError = get_users_producers_accuracy(ref, pred, strata_class, strataDict, assessment_classes)

    # Format accuracies for table
    users_acc = (np.array([list(users.values())])*100).T
    producers_acc = (np.array(list(producers.values()))*100)
    producers_acc = np.append(producers_acc,overall_accuracy*100)
    
    # Append accuracies
    cm = np.concatenate((cm,users_acc),axis=1)
    cm = np.concatenate((cm,[producers_acc]),axis=0)#.astype(int)#.astype(str)
    
    # Handle nan
    # cm[cm==-2147483648] = nan_value

    # Convert to dataFrame
    cm_df = pd.DataFrame(cm,columns=col_labels,index = row_labels)

    cm_df = cm_df.round(2)
   
    # Set headers
    col_header=[('Observed',col) for col in cm_df.columns]
    row_header=[('Predicted',col) for col in cm_df.index]
    cm_df.columns=pd.MultiIndex.from_tuples(col_header)
    cm_df.index=pd.MultiIndex.from_tuples(row_header)

    return cm_df
#---------------------------------Learning Curve------------------------
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), scoring = 'f1'):

    fig = plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Samples")
    plt.ylabel("Score")

    with parallel_backend('threading'):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring, shuffle = True)
    print('train sizes: ')
    print(train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt, fig

#---------------------------------Thresholds------------------------
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, model):
    fig = plt.figure()
    plt.title(model+' Threshold vs. Precision/Recall')
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.ylim([0,1])
    return plt, fig