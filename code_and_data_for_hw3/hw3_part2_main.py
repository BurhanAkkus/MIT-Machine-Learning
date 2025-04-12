import pdb
import numpy as np
from sklearn.model_selection import cross_validate

import code_for_hw3_part2 as hw3
from code_and_data_for_hw3.code_for_hw3_part1 import perceptron,score,y
from code_and_data_for_hw3.code_for_hw3_part2 import averaged_perceptron, xval_learning_alg, reverse_dict


def cross_validation_dataset(data,labels,cross_factor):
    indices = np.random.permutation(data.shape[1])
    data_shuffled = data[:, indices]
    labels_shuffled = labels[:, indices]
    num_examples = data_shuffled.shape[1]  # 391 örnek

    # Test veri sayısını belirleriz. Bu değeri tamsayıya çeviriyoruz.
    num_test = int(num_examples * cross_factor)

    # Eğitim (train) veri sayısı: kalan örnekler
    num_train = num_examples - num_test

    # Eğitim ve test verilerini ayırıyoruz:
    data_train = data_shuffled[:, :num_train]
    label_train = labels_shuffled[:, :num_train]
    data_test = data_shuffled[:, num_train:]
    label_test = labels_shuffled[:, num_train:]
    return data_train,label_train,data_test,label_test

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------
'''
# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

# Construct the standard data and label arrays
auto_data_raw, auto_labels_raw = hw3.auto_data_and_labels(auto_data_all, features)
print('auto data and labels shape', auto_data_raw.shape, auto_labels_raw.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------
features = [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)

if False:
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat)
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()



datasets = [[auto_data_raw, auto_labels_raw], [auto_data,auto_labels]]
Ts = [1,10,50,500]
for dataset in datasets:
    #for t in Ts:
        data_train, label_train,data_test,label_test = cross_validation_dataset(dataset[0],dataset[1],0.10)
        th, th0 = perceptron(data_train,label_train,params={'T':t})
        accuracy1 = int(score(data_test, label_test, th, th0)) / label_test.shape[1]
        th, th0 = averaged_perceptron(data_train,label_train,params={'T':t})
        accuracy2 = int(score(data_test, label_test, th, th0)) / label_test.shape[1]
        print("##############################" , t)
        print("accuracy1: ", accuracy1 , " accuracy2: ", accuracy2)
        accuracy1 = xval_learning_alg(perceptron,dataset[0],dataset[1],10)
        accuracy2 = xval_learning_alg(averaged_perceptron,dataset[0],dataset[1],10)
        print("##############################")
        print("accuracy1: ", accuracy1, " accuracy2: ", accuracy2)
features = [('cylinders', hw3.one_hot),
            #('displacement', hw3.standard),
            #('horsepower', hw3.standard),
            ('weight', hw3.standard),
            #('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            #('origin', hw3.one_hot)
            ]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
dataset = datasets[1]
dataset_ = dataset[1]
data_train, label_train,data_test,label_test = cross_validation_dataset(auto_data, auto_labels, 0.10)
th, th0 = averaged_perceptron(data_train,label_train,params={'T':1})
accuracy1 = int(score(data_test, label_test, th, th0)) / label_test.shape[1]
print(accuracy1)
print(th, th0)

# Your code here to process the auto data

#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)
if False:
    for t in [1,10,50]:
        accuracy1 = xval_learning_alg(perceptron,review_bow_data,review_labels,10, {'T' : t})
        accuracy2 = xval_learning_alg(averaged_perceptron,review_bow_data,review_labels,10, {'T' : t})
        print("##############################")
        print(t," accuracy1: ", accuracy1, " accuracy2: ", accuracy2)

data_train, label_train,data_test,label_test = cross_validation_dataset(review_bow_data, review_labels, 0.10)
th, th0 = averaged_perceptron(data_train,label_train,params={'T':10})
accuracy1 = int(score(data_test, label_test, th, th0)) / label_test.shape[1]
print(accuracy1)
print(th, th0)
top10_indices = np.argsort(th.T)
print(top10_indices[-10:])

index_to_word = reverse_dict(dictionary)
#print(index_to_word)
#print(index_to_word[1])
print([index_to_word[indis] for indis in top10_indices[0][-10:]])

scores = y(review_bow_data,th,th0)
most_positive_index = np.argmax(scores)
most_negative_index = np.argmin(scores)
print("Most positive review:")
print("Index:", most_positive_index)
print("Review Text:", review_texts[most_positive_index])
print("\nMost negative review:")
print("Index:", most_negative_index)
print("Review Text:", review_texts[most_negative_index])
#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# Your code here to process the review data

#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
'''
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    raise Exception("implement me!")

def row_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (m,1) array where each entry is the average of a row
    """
    return np.array(np.sum(x, axis = 1) / x.shape[1]).reshape(-1,1)
    pass



def col_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (n,1) array where each entry is the average of a column
    """
    return np.array(np.sum(x, axis=0) / x.shape[0]).reshape(-1, 1)
    pass


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    n_samples, m, n_cols = x.shape  # Here m = 28, n_cols = 28
    # Compute the average for the top half of each image:
    # Slicing: rows 0 to m//2 (0 to 14, which actually gives rows 0 to 13)
    top_avg = np.mean(x[:, :m // 2, :], axis=(1, 2))  # result shape: (n_samples,)

    # Compute the average for the bottom half of each image:
    # Slicing: rows m//2 to the end (rows 14 to 27)
    bottom_avg = np.mean(x[:, m // 2:, :], axis=(1, 2))  # result shape: (n_samples,)

    # Stack the two computed feature arrays into a (2, n_samples) array.
    return np.vstack((top_avg, bottom_avg))

ans=top_bottom_features(np.array([
    [[1,2,3],[4,5,6]],
    [[3,9,2],[2,1,9]]
]))
print(ans)


# use this function to evaluate accuracy
#acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)

#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data

