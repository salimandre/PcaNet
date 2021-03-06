from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pickle

from pcanet_toolbox import *


###########################################################


N_REDUCED_FEATURES = None # if None no final features nb reduction if an integer then we apply pca reduction on feature map to get N_REDUCED_FEATURES dims
PATH_LFW='/Users/mac/Desktop/python/graph_of_faces/data' # where to find lfw dataset
PATH_FILTERS='/Users/mac/Desktop/python/graph_of_faces/data/filters/model_2/' # where to save filters

###########################################################


lfw_people = fetch_lfw_people(data_home=PATH_LFW, min_faces_per_person=70, resize=0.4)

X_img = lfw_people.images
y = lfw_people.target
target_names = lfw_people.target_names

N_CLASSES = target_names.shape[0]


n_imgs_to_display = 3
for i in range(n_imgs_to_display):
	plt.subplot(1,n_imgs_to_display,i+1)
	plt.axis('off')
	plt.imshow(X_img[i,:,:], cmap=plt.get_cmap('gray'))
	plt.title(target_names[y[i]])
plt.suptitle('lfw dataset')
plt.show()

###########################################################

# build PcaNet model

layer_1 = Layer_PcaNet(n_filters=10, filter_shape=(4,4))
layer_2 = Layer_PcaNet(n_filters=4, filter_shape=(4,4))

model = PcaNet(layers=[layer_1, layer_2], shape_in_blocks = (4,3))

model.summary()

# train PcaNet model

model.train(X_img, n_patches_per_img=500)

# display filters

for layer_name in model.layers.keys():
	model.layers[layer_name].display_filters()

# inference through PcaNet model

outputs = model.inference(X_img)

for i in range(n_imgs_to_display):
	plt.subplot(1,n_imgs_to_display,i+1)
	plt.axis('off')
	I = outputs[2,2,i,:,:]
	I = shift_pixel_values(I, 'int')
	plt.imshow(I, cmap=plt.get_cmap('gray'))
	plt.title(target_names[y[i]])
plt.suptitle('output last layer')
plt.show()

# hashing step

hashed_outputs = model.binary_hashing_step(outputs, sparsity=0.5)

model.display_hashed_outputs([0,1,2], hashed_outputs)

# histogram block step 

feature_map = model.histogram_encoding_step(hashed_outputs)

# features nb reduction

if N_REDUCED_FEATURES is not None:

	pca = PCA(n_components=N_REDUCED_FEATURES, svd_solver='randomized',
	          whiten=True).fit(feature_map)

	print("Projecting the feature_map on ",N_REDUCED_FEATURES,' dims\n')

	feature_map_reduced = pca.transform(feature_map)

	display_feature_map( shift_pixel_values(feature_map_reduced[:n_imgs_to_display,:], 'float'), list_names=[[target_names[y[i]]] for i in range(n_imgs_to_display)], cmap_name='hot')

else:
	display_feature_map(feature_map[:n_imgs_to_display+3,:], list_names=[[target_names[y[i]]] for i in range(n_imgs_to_display+3)], cmap_name='hot')

###########################################################

# features normalization

if N_REDUCED_FEATURES is not None:
	scaler = StandardScaler()
	scaler.fit(feature_map_reduced)
	feature_map = scaler.transform(feature_map_reduced)

	# split into a training and testing set

	X_train, X_test, y_train, y_test = train_test_split(
	    feature_map_reduced, y, test_size=0.25, random_state=42)

else:
	scaler = StandardScaler()
	scaler.fit(feature_map)
	feature_map = scaler.transform(feature_map)

	# split into a training and testing set

	X_train, X_test, y_train, y_test = train_test_split(
	    feature_map, y, test_size=0.25, random_state=42)

###########################################################

# Display data summary

print('\n\n\t *** DATA *** \n')
print("nb total imgs: ".ljust(15), len(X_img))
print("img dims: ".ljust(15), X_img.shape[1:])
print("nb classes: ".ljust(15), N_CLASSES)
print('train: '.ljust(15),X_train.shape, ' ',y_train.shape)
print('test: '.ljust(15),X_test.shape, ' ',y_test.shape)
print('\n\t ************\n\n')


###########################################################

# build a SVM model

clf = SVC(kernel='rbf', class_weight='balanced', C=1e3, gamma=1e-4) #class_weight='balanced'

# training

print('\n\n> training SVM...')

clf = clf.fit(X_train, y_train)

print('done\n')

# inference

y_pred = clf.predict(X_test)

# performance

print(classification_report(y_test, y_pred, target_names=target_names))


###########################################################

# saving filters

with open(PATH_FILTERS+'filters_l1.obj', 'wb') as file:
    pickle.dump(model.layers['layer_1'].filters, file)
with open(PATH_FILTERS+'filters_l2.obj', 'wb') as file:
    pickle.dump(model.layers['layer_2'].filters, file)

with open(PATH_FILTERS+'filters_l1.obj', 'rb') as file:
    filters_l1 = pickle.load(file)
with open(PATH_FILTERS+'filters_l2.obj', 'rb') as file:
    filters_l2 = pickle.load(file)




