from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def knn_classification(train_data,  train_label, test_data,test_label):
    best_res = {}
    best_res['n'] = 0
    best_res['acc'] = 0
    best_res['p_label'] = 0
    best_res['test_label'] = test_label
    best_res['proba'] = 0
    for n in range(2, 30):
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(train_data, train_label)
        p_labels = clf.predict(test_data)
        score = clf.score(test_data, test_label)
        proba = clf.predict_proba(test_data)
        if score > best_res['acc']:
            best_res['acc'] = score
            best_res['n'] = n
            best_res['p_label'] = p_labels
            best_res['proba'] = proba
    return best_res

def svm_classification(train_data, train_label,test_data,  test_label):
    best_res = {}
    best_res['c'] = 0
    best_res['acc'] = 0
    best_res['p_label'] = 0
    best_res['test_label'] = test_label
    for c in range(-10, 10):
        clf = svm.LinearSVC(C=2**c)
        clf.fit(train_data, train_label)
        p_labels = clf.predict(test_data)
        score = clf.score(test_data, test_label)
        decision_value = clf.decision_function(test_data)
        if score >= best_res['acc']:
            best_res['acc'] = score
            best_res['c'] = 2**c
            best_res['p_label'] = p_labels
            best_res['decision_val'] = decision_value
    for c in np.arange(0.1, 20, 0.5):
        clf = svm.LinearSVC(C=c)
        clf.fit(train_data, train_label)
        p_labels = clf.predict(test_data)
        score = clf.score(test_data, test_label)
        decision_value = clf.decision_function(test_data)
        if score >= best_res['acc']:
            best_res['acc'] = score
            best_res['c'] = c
            best_res['p_label'] = p_labels
            best_res['decision_val'] = decision_value
    print(best_res['p_label'])
    return best_res