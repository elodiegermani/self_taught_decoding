from lib import datasets, ae_trainer
from os.path import join as opj
import sys
import getopt
import importlib
import warnings

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=RuntimeWarning)

if __name__ == "__main__":
    data_dir = None
    out_dir = None
    preprocess_type = None
    subset = None
    epochs = None
    model_to_use = None
    batch_size = None
    lr = None

    try:
        OPTIONS, REMAINDER = getopt.getopt(sys.argv[1:], 'o:d:e:b:p:s:m:l:', ['out_dir=', 'data_dir=', 'epochs=', 'batch_size=', 
            'preprocess_type=', 'subset=', 'model=', 'learning_rate='])

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    # Replace variables depending on options
    for opt, arg in OPTIONS:
        if opt in ('-o', '--out_dir'):
            out_dir= arg
        elif opt in ('-d', '--data_dir'):
            data_dir = arg
        elif opt in ('-p', '--preprocess_type'): 
            preprocess_type = str(arg)
        elif opt in ('-s', '--subset'): 
            subset = str(arg)
        elif opt in ('-m', '--model'): 
            model_to_use = str(arg)
        elif opt in ('-e', '--epochs'):
            epochs = int(arg)
        elif opt in ('-b', '--batch_size'):
            batch_size = int(arg)
        elif opt in ('-l', '--learning_rate'):
            lr = float(arg)

    print('OPTIONS   :', OPTIONS)     

    assert(preprocess_type in ['resampled', 'resampled_masked', 'resampled_normalized', 'resampled_masked_normalized'])
    assert(type(batch_size)==int)
    assert(type(epochs)==int)
    assert(type(lr)==float)   

    if data_dir and out_dir and preprocess_type and subset and epochs and batch_size and model_to_use and lr:
        str_lr = "{:.0e}".format(lr)
        package = 'lib.' + model_to_use
        md = importlib.import_module(package)

        train_id_file = opj(data_dir, f"train_{subset}.txt")
        test_id_file = opj(data_dir, f"test_{subset}.txt")
        train_set = datasets.ImageDataset(opj(data_dir, preprocess_type), train_id_file)
        test_set = datasets.ImageDataset(opj(data_dir, preprocess_type), test_id_file)

        model = md.AutoEncoder3D()
        print(f'Training model')
        ae_trainer.trainer(model, train_set, test_set, opj(out_dir, 
                        f"{subset}_maps_{preprocess_type}_{subset}_epochs_{epochs}_batch_size_{batch_size}_{model_to_use}_lr_{str_lr}"),
                         epochs, batch_size, lr)
        
		
