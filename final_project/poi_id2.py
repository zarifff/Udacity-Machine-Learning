#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.metrics
import numpy as np
sys.path.append("../tools/")

#from sklearn.datasets import load_digits
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
#from sklearn.grid_search import GridSearchCV
#from sklearn.model_selection import learning_curve
#from sklearn.cluster import KMeans
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
#from sklearn.decomposition import PCA as sklearnPCA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.datasets.samples_generator import make_blobs

#from pandas.tools.plotting import parallel_coordinates

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'total_payments', 'total_stock_value', 'expenses', 'exercised_stock_options', 'restricted_stock', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### check features

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
import json
with open('enrondata.txt', 'w') as file:
    file.write(json.dumps(data_dict))

file.close()

def scatterPlot(X, Y):
    features = ['poi', X, Y]
    data = featureFormat(data_dict, features)

    for tup in data:
        if tup[0]:
            plt.scatter(tup[1], tup[2], color='red', marker='x')
        else:
            plt.scatter(tup[1], tup[2], color='blue', marker='*')
    
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.grid(True)
    plt.legend(handles=[(mpatches.Patch(color='red', label='POI')), (mpatches.Patch(color='blue', label='Non-POI'))])
    plt.show()


def removeNaNVals():
    #Remove NaN features by replacing with a 0, except for emails
    people_in_database = data_dict.keys()
    keys_per_person = data_dict[people_in_database[0]]

    for person in people_in_database:
        for key in keys_per_person:
            if not key == 'email_address':
                if data_dict[person][key] == 'NaN':
                    data_dict[person][key] = 0


def data_contents():
    #Brief summary of data
    people_in_database = data_dict.keys()
    try: 
        keys_per_person = data_dict[people_in_database[0]]
    except:
        print 'Error! There are no people in the databse'
        pass
    
    print '\nNumber of People in Database: ', len(people_in_database)
    print 'Number of Features per Person: ', len(keys_per_person)

    personOfInterest = 0

    for key, val in data_dict.iteritems():
        if val["poi"] == True:
            personOfInterest += 1   

    print 'Number of Persons of Interest in Database: ', personOfInterest
    print 'Ratio of POI:People = ', personOfInterest/float(len(people_in_database)), '\n'

    
def removeSalaryOutlier():
    salaryList = []
    for key, val in data_dict.iteritems():
        salaryList.append(val['salary'])

    salaryList.sort()
    
    for key, val in data_dict.iteritems():
        if val['salary'] == salaryList[145]:
            print '\nAnomaly Key Name: ', key, '\t...Deleting Key'
            #print val['salary']
            del data_dict[key]
            break

    print 'Anomaly Key Name: THE TRAVEL AGENCY IN THE PARK\t...Deleting Key'
    del data_dict['THE TRAVEL AGENCY IN THE PARK']
    print 'Total number of entries left after removing anomalies: ', len(data_dict)
    
    #for i in range(len(salaryList)):
    #    print salaryList[i]

'''def selectBestFeatures():
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=.60, stratify=labels)

    bestFeatures = SelectKBest(k=8)
    skt = bestFeatures.fit_transform(features_train, labels_train)
    inds = bestFeatures.get_support(True)
    #print bestFeatures.scores_
    print '\nSelecting Best Features...'

    new_features_list = ['poi'] #since poi must be 1st position
    for i in inds:
        print 'Features and score: ', features_list[i+1], bestFeatures.scores_[i]
        new_features_list.append(features_list[i+1])
    
    #return new_features_list

'''

features_list = ['poi', 'bonus', 'total_payments', 'total_stock_value', 'expenses', 'exercised_stock_options', 'to_messages', 'from_this_person_to_poi']


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=.50, stratify=labels)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

def Classifiers():

    clf = KNeighborsClassifier(n_neighbors=8)
    clf.fit(features_train, labels_train)

    print '\nK-Nearest Neighbors R-squared Score: ', clf.score(features_test, labels_test)

    clf = LogisticRegression(C=100.0)
    clf.fit(features_train, labels_train)

    print 'Logistic Regression R-squared Score: ', clf.score(features_test, labels_test)

    clf = AdaBoostClassifier()
    clf.fit(features_train, labels_train)

    print 'Ada Boost R-squared Score: ', clf.score(features_test, labels_test)


data_contents()
removeNaNVals()
removeSalaryOutlier()
#scatterPlot('salary', 'bonus')
scatterPlot('from_this_person_to_poi', 'salary')
Classifiers()


'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)
'''