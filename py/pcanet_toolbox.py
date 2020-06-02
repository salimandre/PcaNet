import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.feature_extraction import image
from tqdm import tqdm


# shift pixel values into [0, 255] interval
def shift_pixel_values(I, dtype):
    maxpv = np.max(I)
    minpv = np.min(I)
    I = 255 * np.divide((I - np.min(I)), (np.max(I)-np.min(I)))
    return np.round(I).astype(dtype)


def split_img_into_blocks(img, n_blocks_row, n_blocks_col):

    m,n = img.shape

    q_m = m//n_blocks_row
    q_n = n//n_blocks_col

    I_blocks = np.zeros((n_blocks_row*n_blocks_col, q_m, q_n))
    for i in range(n_blocks_row):
        for j in range(n_blocks_col):
            #I_blocks += [ img[i*q_m:(i+1)*q_m,j*q_n:(j+1)*q_n].flatten() ]
            I_blocks[i*n_blocks_col+j,:,:] =  img[i*q_m:(i+1)*q_m,j*q_n:(j+1)*q_n]

    return I_blocks

def display_feature_map(feature_map, list_names=None, cmap_name='hot'):

    f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 6]})
    a0.plot()
    a0.axis('off')
    if list_names is not None:
        names = a0.table(cellText=list_names, loc='center', edges='open')
    names.scale(1,5.2-0.3*len(feature_map))
    names.auto_set_font_size(False)
    names.set_fontsize(8)
    a1.imshow(shift_pixel_values(np.log(feature_map+2), 'int'), cmap=plt.get_cmap(cmap_name), interpolation='nearest', aspect='auto')
    a1.axis('off')
    a1.set_title('feature map')
    plt.show()

class PcaNet:

    def __init__(self, layers=None, shape_in_blocks = (3,3)): 

        self.shape_in_blocks = shape_in_blocks

        self.layers={}
        self.raw_n_features = 1
        self.n_bits = 1
        for i, layer in enumerate(layers):
            self.layers['layer_'+str(i+1)]=layer
            if i+1<len(layers):
                self.raw_n_features = self.raw_n_features * layer.n_filters
            else:
                self.raw_n_features = self.raw_n_features * np.prod(self.shape_in_blocks) * 2**layer.n_filters
            if i>0:
                self.n_bits = self.n_bits * layer.n_filters

    def summary(self):

        ljust_param = 40
        print('\n---------------------------------------- PcaNet ----------------------------------------')
        print('----------------------------------------------------------------------------------------')
        print('Layer (type)'.ljust(ljust_param,' ')+'Output Shape'.ljust(ljust_param,' ')+'Params #'.ljust(ljust_param,' '))
        print('----------------------------------------------------------------------------------------')
        shape_temp = 'n_imgs, height, width)'
        for i in range(len(self.layers)):
            layer_name = 'layer_'+str(i+1)
            layer = self.layers[layer_name]

            str_layer_type = 'Conv2D '+str(layer.filter_shape)

            shape_temp = str(layer.n_filters)+', '+shape_temp
            str_output_shape = '('+shape_temp

            str_params = str(layer.n_filters*np.prod(layer.filter_shape))

            str_row = str_layer_type.ljust(ljust_param,' ')+str_output_shape.ljust(ljust_param,' ')+str_params.ljust(ljust_param,' ')
            print(str_row)


        str_layer_type = 'Hashing'
        n_filters_l1 = self.layers['layer_1'].n_filters
        str_output_shape = '(n_imgs, '+str(n_filters_l1)+', height, width)'
        str_params = '0'
        str_row = str_layer_type.ljust(ljust_param,' ')+str_output_shape.ljust(ljust_param,' ')+str_params.ljust(ljust_param,' ')
        print(str_row)

        str_layer_type = 'Histogram Blocks '+str(self.shape_in_blocks)
        str_output_shape = '(n_imgs, '+str(self.raw_n_features)+')'
        str_params = '0'
        str_row = str_layer_type.ljust(ljust_param,' ')+str_output_shape.ljust(ljust_param,' ')+str_params.ljust(ljust_param,' ')
        print(str_row)

        print('----------------------------------------------------------------------------------------\n')


    def train(self, inputs, n_patches_per_img=100):

        print('\n\n> training PcaNet...\n')

        for i in range(len(self.layers)):

            layer = self.layers['layer_'+str(i+1)]

            print('\n\n\t> layer_'+str(i+1)+':\n')

            layer.sample_patches_from_imgs(inputs, n_patches_per_img)

            layer.train_filters()

            layer.clear_patches()

            inputs = layer.inference(inputs)

        print('> training PcaNet... Done!\n')


    def inference(self, inputs):

        print('\n\n> inference through PcaNet...\n')

        for i in range(len(self.layers)):
            layer_name = 'layer_'+str(i+1)
            layer = self.layers[layer_name]
            print('\n\n\t> '+layer_name+':\n')
            inputs = layer.inference(inputs)

        print('> inference through PcaNet... Done!\n')

        return inputs

    def binary_hashing_step(self, inputs, sparsity):

        # inputs = outputs from inference through PcaNet 
        # of dims: n_filters_li * n_filters_li-1 * ... * n_filters_l1 * n_imgs * height * width

        print('> computing binary hashing...')

        inputs_shape = inputs.shape
        h, w = inputs.shape[-2:]
        n_imgs = inputs.shape[-3]
        nb_bits = np.prod(inputs.shape[:-4])
        n_filters_l1 = self.layers['layer_1'].n_filters

        hashed_outputs = np.zeros((n_imgs,n_filters_l1,h,w))

        inputs = np.reshape(inputs, (n_filters_l1, nb_bits, n_imgs, h, w))

        for i in tqdm(range(n_imgs)):
            T_i = np.zeros((n_filters_l1, h, w))
            for j in range(self.layers['layer_1'].n_filters):
                O_ij = np.zeros((h,w))
                for k in range(nb_bits):
                    O_ijk = inputs[j,k,i,:,:]
                    q_thresh = np.quantile(O_ijk, sparsity)
                    O_ij += 2**k * np.floor(O_ijk-q_thresh).clip(0, 1)
                T_i[j] = O_ij
            hashed_outputs[i] = T_i

        print('> computing binary hashing... Done!\n')

        return hashed_outputs

    def display_hashed_outputs(self, list_id_imgs, hashed_outputs, cmap_name='gray'):
        # outputs = tensor of shape: n_imgs * n_filters_1 * height * width

        n_filters_l1=self.layers['layer_1'].n_filters
        # proper n_cols in plot with grid plot of shape (2, n_cols)
        n_cols = [(i, 2*i-n_filters_l1) for i in range(10) if 2*i-n_filters_l1>=0][0][0]

        for k in list_id_imgs:

            for j in range(n_filters_l1):
                I_hashed = hashed_outputs[k,j,:,:]
                I_hashed = shift_pixel_values(I_hashed, 'int')
                plt.subplot(2,n_cols,j+1)
                plt.axis('off')
                plt.imshow(I_hashed, cmap=plt.get_cmap(cmap_name))

            plt.suptitle('output after hashing step')
            plt.show()

    def histogram_encoding_step(self, inputs):

        # inputs = hashed outputs from inference of dims: n_imgs * n_filters_l1 * height * width

        print('\n\n> computing feature map from PCANet...')

        inputs_shape = inputs.shape
        n_imgs = inputs.shape[0]
        n_filters_l1 = self.layers['layer_1'].n_filters

        feature_map = np.zeros((n_imgs, self.raw_n_features))
        feature_row=[]

        for i in tqdm(range(n_imgs)):

            for j in range(n_filters_l1):

                I_blocks = split_img_into_blocks(inputs[i,j,:,:], self.shape_in_blocks[0], self.shape_in_blocks[1])

                for b in I_blocks:
                    feature_row += np.histogram(b, bins=2**self.n_bits)[0].tolist()

            feature_map[i,:] = feature_row
            feature_row=[]

        print('\n\n> computing feature map from PCANet... Done!\n')

        return feature_map



class Layer_PcaNet:

    def __init__(self, filters=None, n_filters=None, filter_shape=None):

        self.filters = filters

        if self.filters is not None:
            self.n_filters = len(self.filters)
            self.filter_shape = self.filters[0].shape

        elif n_filters is not None and filter_shape is not None:
            self.n_filters = n_filters
            self.filter_shape = filter_shape
            self.filters = [np.zeros(self.filter_shape)]*self.n_filters

        else:
            raise ValueError("all args are None! You must have either 'filters' to be not None or have both 'n_filters' and 'filter_shape' to be not None.")

        self.bag_of_patches = None

    def sample_patches_from_imgs(self, inputs, n_patches_per_img, seed=None):

        # inputs = array/list of length n_imgs with imgs
        # n_patches_per_img = max number (integer) of patches sampled per img

        n_imgs = np.prod(inputs.shape[:-2])
        h, w = inputs.shape[-2:]

        inputs = np.reshape(inputs, (n_imgs, h, w))

        print('\t\tsampling patches for layer ...')

        bag_of_patches= np.zeros((n_imgs*n_patches_per_img, self.filter_shape[0], self.filter_shape[1]))

        for i, img in enumerate(tqdm(inputs)):
            patches = image.extract_patches_2d(img, self.filter_shape, max_patches=n_patches_per_img, random_state=seed)
            mu = np.mean(patches)
            patches = patches - mu
            bag_of_patches[i*n_patches_per_img:(i+1)*n_patches_per_img,:,:] = patches

        print('\t\tdone\n')

        self.bag_of_patches = bag_of_patches

    def train_filters(self):

        print('\t\ttraining filters from layer ...')

        # reshape bag of patches from 3D tensor into 2D matrix
        X = np.reshape(self.bag_of_patches, (len(self.bag_of_patches),np.prod(self.filter_shape)))
        X = np.int64(X)

        # Sample Covariance Matrix between pixels
        Cov = X.T@X

        #eigen values in descending order
        ev, U = np.linalg.eigh(Cov)

        # select L1 principal vectors to get an orthogonal basis of size L1 in patch space (dim=k1*k2)
        top_eigvec = U[:,-self.n_filters:].T

        # build eigenvectors as filters of shape (k1, k2) 
        new_filters = []
        for v in top_eigvec:
            new_filters += [np.reshape(v, self.filter_shape)]

        print('\t\tdone\n')

        self.filters = new_filters


    def clear_patches(self):
        self.bag_of_patches = None


    def display_filters(self, cmap_name='gray'):

        # cmap_name = 'gray' 'gist_gray' 'Blues' 'plasma_r' 'autumn' ...

        # compute for 2 rows proper nb of columns to display in plot
        n_cols = [(i, 2*i-self.n_filters) for i in range(10) if 2*i-self.n_filters>=0][0][0]
        
        for i in range(self.n_filters):
            plt.subplot(2,n_cols,i+1)
            plt.axis('off')
            plt.imshow(self.filters[i], cmap=plt.get_cmap(cmap_name))

        plt.suptitle(str(self.n_filters)+' filters from PcaNet layer')
        plt.show()


    def inference(self, inputs):

        # inputs = array/list of length n_inputs with imgs

        print('\t\tinference through layer...')

        inputs_shape = inputs.shape

        outputs_shape = tuple([self.n_filters]+list(inputs.shape))
        outputs = np.zeros(outputs_shape)

        n_imgs = np.prod(inputs.shape[:-2])
        h, w = inputs.shape[-2:]

        inputs = np.reshape(inputs, (n_imgs, h, w))

        for j in range(self.n_filters):
            outputs_j = []
            for i in range(n_imgs):
                outputs_j += [cv2.filter2D(inputs[i], ddepth=-1, kernel=self.filters[j])]

            outputs_j = np.reshape(outputs_j, inputs_shape)
            outputs[j] = outputs_j

        print('\t\tdone\n')

        return np.array(outputs)

# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------






