import os
import time
import argparse
import pandas as pd
from competitions.tools.timer import timer
from competitions.tools.DataReader import DataReader


def train_and_eval(X_train, y_train, X_val, y_val, module):
    """
    train model and eval hold-out performance
    BTW, write scores to csv files

    Parameters
    ----------
    X_train, y_train, X_val, y_val: features and targets

    module: a python module

    Return
    ------
    training logs
    """
    # get model
    model = module.get_model()
    # train model
    print('Training model...')
    model = model.train(X_train, y_train, X_val, y_val)
    best_param = model.best_param
    best_score = model.best_score
    print("Best param: {:.4f} with best score: {}".format(best_param, best_score))  # noqa
    return pd.DataFrame({'best_param': [best_param], 'best_score': [best_score]})  # noqa


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Quora Insincere Questions Classification",
        description="Run Model Validation and Pick the Best Best Param")
    parser.add_argument('--datapath', nargs='?', default=os.environ['DATA_PATH'],
                        help='input data path')
    parser.add_argument('--model', nargs='?', default='NeuralNetwork',
                        help='model version')
    return parser.parse_args()


if __name__ == '__main__':
    # config
    RANDOM_STATE = 2
    SHUFFLE = True
    TEST_SIZE = 0.8
    # get args
    args = parse_args()
    datapath = args.datapath
    model = args.model

    t0 = time.time()
    # 1. import module
    module = __import__(model)

    dr = DataReader(os.path.join(datapath, 'quora', 'train.csv'), module)

    with timer("Load and Preprocess"):
        X_t, X_v, y_t, y_v = dr.get_split(TEST_SIZE)

    with timer('Training and Tuning'):
        df_score = train_and_eval(X_t, y_t, X_v, y_v, module)
        filepath = os.path.join(datapath, 'trainer_{}.csv'.format(model))
        df_score.to_csv(filepath)
        print('Save CV score file to {}'.format(filepath))

    print('Entire program is done and it took {:.2f}s'.format(time.time() - t0))  # noqa
