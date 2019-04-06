"""
 This code is released upon submission to USENIX Security 2018
 Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning

 This code implements Stacked Denoising Autoencoders based on
 1. Theoretical concept proposed by Vincent et al.
 2. Hyperparameter tuning and model selection are based to Abe et al. and what have tested
    2.1 SDAE is data-specific DNN, Hyperparameter tuning nees to be done with different datasets
        to gain effective result.
    2.2 Grid searching or semi-grid searching, or extended searching could be performed for 2.1
 3. The denoising is applied by the use of greedy layerwise training proposed by Benio et al.
    (apply denoising process only for pre-training)
 4. The code is written by Keras with Tensorflow as the backend to make it simply understandable

 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
   - K. Abe and S. Goto. Fingerprinting Attack on Tor Anonymity using
   Deep Learning. in Proceedings of the APAN, 2016.

"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.optimizers import Adamax
from keras.layers.advanced_activations import ELU
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import time
import sys
import os

from timeit import default_timer as timer
from shutil import copyfile

from data_prep import LoadDataMon_Large

# Initial parameters
# This part can be additonally applied with command-line arguments
# Length of input features = len([+1,-1,+1,-1])
FEATURE_LENGTH = 5000

# Number of Classes
NB_CLASSES = 10

# Number of hidden units for each hidden layer
# e.g. [number of hidden units for 1st Hidden Layer,... ]
HIDDEN_LAYERS = [2500, 1000, NB_CLASSES]

# Number of input & output for each denoising layers
DAE_INPUT = [FEATURE_LENGTH] + HIDDEN_LAYERS

# Number Epochs for pre-training (pre-train each layer to learn representations of input)
NUM_PRE_EPOCH = 3
PRE_EPOCH = NUM_PRE_EPOCH     # Pre-training epochs
# Number Epochs for fine tuning (training classification model process)
FINE_EPOCH = 3

# Batch Size for pre-trainin process
BATCH_SIZE_PRE = 64
# Batch Size for fine-tuning process
BATCH_SIZE_FINE = 128

# VERBOSE
VERBOSE = 2     # Print only summary of each epoch

# Optimizer for pre-training process
OPTIMIZER_PRE = Adamax(lr=0.002, beta_1=0.9, beta_2=0.998, epsilon=1e-08, decay=0.0)
# Optimizer for fine-tuning process
OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.998, epsilon=1e-08, decay=0.0)

TIMEDETAIL = str(time.strftime("%Y-%m-%d-%H:%M"))
PATH = ""


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# /Use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Use CPU/ Delete these two lines if you are using CUDA


class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "wb")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message.encode())

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

    def close(self):
        self.log.close()


def setup_logging():
    current_path = os.getcwd()

    # Loging the file in the folder /Temp_Result/
    # The log file will copy the code used for SDAE and its result
    PATH = "{}/Temp_Result/{}/".format(current_path, TIMEDETAIL)
    os.makedirs(PATH)
    outfile = "{}result_{}.txt".format(PATH, TIMEDETAIL)

    predicted_result = "{}predicted_{}.txt".format(PATH, TIMEDETAIL)
    predicted_result_exp = "{}predicted_{}.csv".format(PATH, TIMEDETAIL)

    src = "{}/{}".format(current_path, os.path.basename(__file__))
    dst = "{}code_{}.txt".format(PATH, TIMEDETAIL)
    copyfile(src, dst)
    sys.stdout = Logger(outfile)


def add_corruption(corruption_noise, X_train):
    corruption_array = np.random.choice([0.0, 1.0], size=X_train.shape, p=[corruption_noise, 1-corruption_noise])

    X_noised_train = np.multiply(X_train, corruption_array)

    return X_noised_train


def load_data():
    # Load data

    # If you want to small dataset use the two lines of code below
    # and comment the next two lines
    """
    from data_prep import LoadDataMon_Small
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = LoadDataMon_Small()
    """
    # If you want to large dataset use the two lines of code below
    # and comment the previous two lines

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = LoadDataMon_Large()

    K.set_image_dim_ordering("tf") # tf is tensorflow
    # Convert data to Float32 to be best executable in GPU
    X_train = X_train.astype('float32')
    Y_train = Y_train.astype('float32')
    X_valid = X_valid.astype('float32')
    Y_valid = Y_valid.astype('float32')
    X_test = X_test.astype('float32')
    Y_test = Y_test.astype('float32')

    # One-hot encoding (Convert each label into binary ([0,1]) vectors)
    # e.g. class 1 in convert to ([[0,1,0,...,0],....])
    Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
    Y_valid = np_utils.to_categorical(Y_valid, NB_CLASSES)
    Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def pre_train(X_train):
    # =====================================================================================
    # Pretraining Process
    # =====================================================================================
    # Initial X vector
    X_input = X_train
    print("Shape of X_input: {}".format(np.array(X_input).shape))
    
    # Weights after done pre-training on each layer
    # it is used for fine-tuning process later
    WeightBias = {}

    # Denoising Autoencoders are being stacked one by one
    # Corruption Level
    # e.g. [1st Hidden Layer's corruption, 2nd Hidden Layer's corruption,...]
    CORRUPTION_LEVEL = [0.3,0.3,0.3,0.3]
    print("Corruption Level: {}".format(CORRUPTION_LEVEL))

    for h in range(len(HIDDEN_LAYERS)):
        # An example x is stochastically corrupted
        corruption_noise = CORRUPTION_LEVEL[h]
        X_noised_train = add_corruption(corruption_noise, X_input)

        # Parameters for each pre-training layer
        hidden_unit = DAE_INPUT[h + 1]
        hidden_input_dim = DAE_INPUT[h]
        hidden_output_dim = DAE_INPUT[h]

        print("Pre-trainig for Denoising Layer : {}".format(h+1))
        print("Number of Hidden Units : {}".format(hidden_unit))
        print("--- Input Dimension  : {}".format(hidden_input_dim))
        print("--- Output Dimension : {}".format(hidden_output_dim))

        DAE_layer = Sequential()
        DAE_layer.add(Dense(hidden_unit, input_dim=hidden_input_dim,
                            name='Pretrain_hidden_encode_{}'.format(h+1)))

        if h == 0: 
            # For first pre-trainin layer
            DAE_layer.add(ELU(alpha=1.0, name='Pretrain_activation_encode_{}'.format(h+1)))

        else:
            # For the next following layers
            DAE_layer.add(Activation('tanh', name='Pretrain_activation_encode_{}'.format(h + 1)))
        
        DAE_layer.add(Dense(hidden_output_dim, name='Pretrain_hidden_decode_{}'.format(h+1)))

        # Reconstruction to from corrupted input to the original input
        # This process allow each layers to learn the representation of the input in each layer.
        DAE_layer.compile(optimizer=OPTIMIZER_PRE, loss='mse')
        DAE_layer.fit(X_noised_train, X_input, epochs=PRE_EPOCH, batch_size=BATCH_SIZE_PRE, verbose=VERBOSE)


        # Extract configuration value and weights of hidden layer
        # They include both encoding and decoding bias and weights
        # Encoding happens when we feed X through the hidden layer
        # Decoding happens when we try to reconstruct bact to the original X

        Weights_data = DAE_layer.get_weights()

        Encoding_Weight = Weights_data[0]   # (Xn, #Hidden units) e.g. (5000, 1000)
        Encoding_Bias = Weights_data[1]       # (#Hidden unit,) e.g. (1000,)

        # Create weights and bias information for fine-tuning process
        WeightBias[h+1] = [(Encoding_Weight), (Encoding_Bias)]

        # Get shape from data
        [Xshape, Yshape] = np.shape(X_input.transpose())  # (Xi-n, Instances)

        # transpose weights matrix into (Hidden Units, Xi-n)
        # -- Each Xi column, contains vector of weights to which Xi connected
        Encoding_Weight_Trans = Encoding_Weight.transpose()
        Encoding_Bias = np.outer(Encoding_Bias, np.ones((1, Yshape), dtype=np.float))


        # Activation Computation

        # Compute pre-training activation of Encoding process including Weights and Biases
        # This computation is as described above (Activation Computation)
        Encoding_Pre_Act = np.matmul(Encoding_Weight_Trans, X_input.transpose()) + Encoding_Bias

        # Then peform Activation by applying non-linearlity to Encoding_Pre_Act

        if h == 0:
            Encoding_Activation = np.where(Encoding_Pre_Act<0, (1.0 * (np.exp(Encoding_Pre_Act) - 1.)), Encoding_Pre_Act)
        else:
            Encoding_Activation = np.maximum(Encoding_Pre_Act, 0, Encoding_Pre_Act)


        # X_input for the next pre-training layer
        # (Instances, NumberHiddenUnit of this layer will be the next input)
        X_input = Encoding_Activation.transpose()
        print("Shape of X_input: {}".format(np.array(X_input).shape))
    
    return WeightBias, DAE_layer


def fine_tune(WeightBias, X_train, X_valid, Y_train, Y_valid):
    # =====================================================================================
    # Fine-Tuning Process
    # =====================================================================================
    Finetune = Sequential()
    drop_out_value = [0.6, 0.4, 0.2, 0.1]
    print("Drop out value : {}".format(drop_out_value))
    for d in range(len(HIDDEN_LAYERS)):

        hidden_unit = DAE_INPUT[d + 1]
        hidden_input_dim = DAE_INPUT[d]
        hidden_output_dim = DAE_INPUT[d+1]

        # Build DNN layer by layer
        if d == 0:
            Finetune.add(Dense(hidden_unit, input_dim=hidden_input_dim,
                            name = 'Finetune_hidden_{}'.format(d+1)))

            print("Create Layer {} with {} hidden units".format(d + 1, hidden_unit))
            #Finetune.add(ELU(alpha=1.0, name='Finetune_activation_%s'%str(d+1)))
            Finetune.add(ELU(alpha = 1.0, name='Finetune_activation_{}'.format(d + 1)))
            Finetune.add(Dropout(drop_out_value[d], name = 'Finetune_dropout_{}'.format(d+1)))
            current_layer = d
            Finetune.layers[current_layer].set_weights(WeightBias[d + 1])
            current_layer = current_layer + 3

        else:
            Finetune.add(Dense(hidden_unit, name = 'Finetune_hidden_{}'.format(d+1)))
            print("Create Layer {} with {} hidden units".format(d + 1, hidden_unit))

            if (d + 1) != len(HIDDEN_LAYERS):
                Finetune.add(Activation('tanh', name = 'Finetune_activation_{}'.format(d+1)))
                Finetune.add(Dropout(drop_out_value[d], name = 'Finetune_dropout_{}'.format(d+1)))
                Finetune.layers[current_layer].set_weights(WeightBias[d + 1])
                current_layer = current_layer + 3
            
            else:
                Finetune.add(Activation('softmax', name="softmax"))
                Finetune.layers[current_layer].set_weights(WeightBias[d + 1])




    #Finetune.add(Dense(NB_CLASSES, kernel_initializer = glorot_uniform(seed=0),name='Dense_classes'))

    Finetune.compile(
        loss='categorical_crossentropy',
        optimizer=OPTIMIZER,
        metrics=['accuracy']
    )

    history = Finetune.fit(
        X_train, 
        Y_train, 
        epochs=FINE_EPOCH, 
        batch_size=BATCH_SIZE_FINE,
        validation_data=(X_valid, Y_valid),
        verbose=VERBOSE
    )

    return Finetune, history


def evaluate(Finetune, X_test, Y_test):
    print("Start evaluating a classifier")
    score = Finetune.evaluate(X_test, Y_test, verbose=VERBOSE)
    print("Test Accuracy: {}".format(score[1]))
    return score


def print_results(history, DAE_layer, Finetune, score):
    print(DAE_layer.summary())
    print(Finetune.summary())

    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    train_acc = "Accuracy_{}.pdf".format(TIMEDETAIL)
    plt.savefig(PATH + train_acc)
    plt.gcf().clear()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    valid_acc = "Loss_{}.pdf".format(TIMEDETAIL)
    plt.savefig(PATH + valid_acc)
    plt.gcf().clear()
    sys.stdout.close()



if __name__ == '__main__':
    print("Number of pre-training epochs: {}".format(NUM_PRE_EPOCH))
    print("Number of Fine-tuning epochs: {}".format(FINE_EPOCH))

    setup_logging()

    load_start = timer()
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_data()
    load_end = timer()
    print('{} Training samples'.format(X_train.shape[0]))
    print('{} Validation samples'.format(X_valid.shape[0]))
    print('{} Test samples'.format(X_test.shape[0]))
    print("\nLoading Data Done! : {:.2f} s".format(load_end - load_start))

    pretrain_start = timer()
    WeightBias, DAE_layer = pre_train(X_train)
    pretrain_end = timer()
    print("Pretraining Done! : {:.2f} s".format(pretrain_end - pretrain_start))

    train_start = timer()
    Finetune, history = fine_tune(WeightBias, X_train, X_valid, Y_train, Y_valid)
    train_end = timer()
    print("Finetuning Done! : {:.2f} s".format(train_end - train_start))

    test_start = timer()
    score = evaluate(Finetune, X_test, Y_test)
    test_end = timer()
    print("Testing Done! : {:.2f} s".format(test_end - test_start))
    print("\nTest score: {}".format(score[0]))
    print("Test accuracy: {}".format(score[1]))

    print_results(history, DAE_layer, Finetune, score)