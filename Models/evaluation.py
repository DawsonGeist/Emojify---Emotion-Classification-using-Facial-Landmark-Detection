import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics 
import math

def convert_y_to_binary(desired_class,y,y_pred):
    bin_y = []
    bin_y_pred = []
    for count, result in enumerate(y):
        if result == desired_class:
            bin_y.append(1)
        else:
            bin_y.append(0)
        if y_pred[count] == desired_class:
            bin_y_pred.append(1)
        else:
            bin_y_pred.append(0)
    return bin_y, bin_y_pred

def get_roc_for_all(classes, y_test, y_pred):
    for label in classes:
        # y2_test, y2_pred = convert_y_to_binary(label, y_test, y_pred)
        fpr, tpr, thresholds =roc_curve(y_test, y_pred[:,label], pos_label=label)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
        display.plot()
        plt.show()


# Binarize
# def binarize(model, X_train, X_test, y_train, y_test):
#     label_binarizer = LabelBinarizer().fit(y_train)
#     y_onehot_test = label_binarizer.transform(y_test)
#     y_score = model.fit(X_train, y_train).predict_proba(y_test)
#     return y_onehot_test, y_score

# # Plot ROC curve
# def roc_curve(models, onehot_tests, scores):
#     # plotting parameters
#     cols = 3
#     linewidth = 1
#     rows = math.ceil(7 / cols)
#     emotions = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}

#     fig, axs = plt.subplots(rows, cols, figsize=(15, rows*6))
#     model_names = {0:'SVM', 1:'RF', 2:'DeepFace'}

#     for i in range(7):
#         emotion_name = emotions[i]
#         for m in range(len(models)):
#             model_name = model_names[m]
#             display = RocCurveDisplay.from_predictions(onehot_tests[m][:, i],
#                                                        scores[m][:, i],
#                                                        name=f"{model_name}",
#                                                        linewidth=linewidth,
#                                                        ax=axs[i // cols, i % cols],)

#         axs[i // cols, i % cols].plot([0, 1], [0, 1], linewidth=linewidth, linestyle=":")
#         axs[i // cols, i % cols].set_title(emotion_name)
#         axs[i // cols, i % cols].set_xlabel("False Positive Rate")
#         axs[i // cols, i % cols].set_ylabel("True Positive Rate")
#     plt.tight_layout(pad=2.0)  # spacing between subplots
#     plt.show()


def deepface_comparison():
    y_test = np.loadtxt('y_true_full.csv', delimiter=',')
    y_pred = np.loadtxt('deepface_pred_full.csv', delimiter=',')

    print(accuracy_score(y_test, y_pred))
    CNN_avg_precisionR = precision_score(y_test, y_pred, average='weighted')
    CNN_recallR = recall_score(y_test, y_pred, average='weighted')
    CNN_f1R = f1_score(y_test, y_pred, average='weighted')
    print(f'Average precision for DF: {CNN_avg_precisionR}')
    print(f'Recall for DF: {CNN_recallR}')
    print(f'F1-measure for DF: {CNN_f1R}')
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    y2_test, y2_pred = convert_y_to_binary(1, y_test, y_pred)
    cm = confusion_matrix(y2_test, y2_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    y_pred = np.loadtxt('deepface_pred_proba_full.csv', delimiter=',')
    get_roc_for_all([0,1,2,3,4,5,6], y_test, y_pred)
    return y_pred

def svm_comparison(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(accuracy_score(y_test, y_pred))
    CNN_recallR = recall_score(y_test, y_pred, average='weighted')
    CNN_f1R = f1_score(y_test, y_pred, average='weighted')

    # y_score =  clf.decision_function(X_test)
    CNN_avg_precisionR = precision_score(y_test, y_pred, average='weighted')

    print(f'Average precision for SVM: {CNN_avg_precisionR}')
    print(f'Recall for SVM: {CNN_recallR}')
    print(f'F1-measure for SVM: {CNN_f1R}')

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    y2_test, y2_pred = convert_y_to_binary(1, y_test, y_pred)
    cm = confusion_matrix(y2_test, y2_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    y_pred = clf.decision_function(X_test)

    get_roc_for_all([0,1,2,3,4,5,6], y_test, y_pred)
    return clf, y_pred

def random_forest_comparison(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    y2_test, y2_pred = convert_y_to_binary(1, y_test, y_pred)
    cm = confusion_matrix(y2_test, y2_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    get_roc_for_all([0,1,2,3,4,5,6], y_test, y_pred)


def data_analysis(x,y):
    c = np.cov(x.T)
    e_vals, e_vecs = np.linalg.eig(c)
    print(c)
    print(e_vals, e_vecs)
    label_count = {
        'Angry' : 0,
        'Disgust' : 0,
        'Fear' : 0,
        'Happy' : 0,
        'Sad' : 0,
        'Surprise' : 0,
        'Neutral' : 0
    }
    for label in y:
        if label == 0:
            label_count['Angry'] += 1
        elif label == 1:
            label_count['Disgust'] += 1
        elif label == 2:
            label_count['Fear'] += 1
        elif label == 3:
            label_count['Happy'] += 1
        elif label == 4:
            label_count['Sad'] += 1
        elif label == 5:
            label_count['Surprise'] += 1
        elif label == 6:
            label_count['Neutral'] += 1
    
    for result in label_count.keys():
        print(result, ": ", label_count[result])

def svm_predict(model, features):
    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    y_pred = model.predict(features.reshape((1,18)))
    return objects[int(y_pred[0])]

def get_svm():
    x = np.loadtxt('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Data\\feature_matrix.csv', delimiter=',')
    y = np.loadtxt('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Data\\feature_matrix_classes.csv', delimiter=',')
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(x, y)
    return clf

def convert_y(y):
    y_expanded = []
    for label in y:
        if label == 0:
            y_expanded.append([1,0,0,0,0,0,0])
        elif label == 1:
            y_expanded.append([0,1,0,0,0,0,0])
        elif label == 2:
            y_expanded.append([0,0,1,0,0,0,0])
        elif label == 3:
            y_expanded.append([0,0,0,1,0,0,0])
        elif label == 4:
            y_expanded.append([0,0,0,0,1,0,0])
        elif label == 5:
            y_expanded.append([0,0,0,0,0,1,0])
        elif label == 6:
            y_expanded.append([0,0,0,0,0,0,1])
    return y_expanded


def main():
    x = np.loadtxt('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Data\\feature_matrix.csv', delimiter=',')
    y = np.loadtxt('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Data\\feature_matrix_classes.csv', delimiter=',')
    # svm_comparison(x,y)
    deepface_comparison()
    # random_forest_comparison(x,y)
    # data_analysis(x,y)
    
if __name__ == '__main__':
    main()



