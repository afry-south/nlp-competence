import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from competitions.tools.timer import timer


class DataReader(object):
    def __init__(self,
                 train_file,
                 module,
                 test_file=None):

        if not train_file:
            raise Exception("DataReader requires a train_file!")
        if not module:
            raise Exception("DataReader requires a model that can transform data!")
        self.raw_test = None
        if test_file:
            print("Loading test_data (%s) into dataframe" % test_file)
            self.test_data = pd.read_csv(test_file)
            self.raw_test = self.test_data[['question_text']]
            print("Test data with shape: ", self.test_data.shape)

        print("Loading train_data (%s) into dataframes" % train_file)
        self.train_data = pd.read_csv(train_file)
        self.raw_train = self.train_data[['question_text']]
        print("Train data with shape: ", self.train_data.shape)
        train_test_cut = self.train_data.shape[0]
        if isinstance(self.raw_test, pd.DataFrame):
            df_all = pd.concat([self.raw_train, self.raw_test],
                               axis=0).reset_index(drop=True)
        else:
            df_all = self.raw_train
        self.df_all = df_all


        print("Transforming the data")
        with timer('Transforming data'):
            if module:
                X_features = module.transform(df_all['question_text'])
            else:
                X_features = df_all['question_text']
            # Multiple Inputs
            if isinstance(X_features, list):
                self.X_train = [X[:train_test_cut] for X in X_features]
                self.X_test = [X[train_test_cut:] for X in X_features]
            else:
                self.X_train = X_features[:train_test_cut]
                self.X_test = X_features[train_test_cut:]

    def get_split(self, split=0.8, random_state=2, shuffle_data=True):
        """
        :param split: float - % to be training data
        :param random_state: int - init_state for random to keep random stale
        :param shuffle_data: if to shuffle
        :return: X_t, X_v, y_t, y_v where X = training and Y = validation.
        t = training data & v = class
        """
        print("Creating validation data by splitting (%s)" % split)
        train_data = self.train_data
        X_train = self.X_train

        X_t, X_v, y_t, y_v = train_test_split(
            X_train, train_data.target,
            test_size=(1 - split), random_state=random_state,
            shuffle=shuffle_data, stratify=train_data.target)

        return X_t, X_v, y_t, y_v

    def get_kfold(self, k=5, shuffle_data=True, random_state=2):
        """
        :param k: int - Number of folds.
        :param shuffle_data: boolean - If we should shuffle
        :param random_state: int - init_state for random to keep random stale
        :return: a generator that yields the folds.
        """
        print("Creating validation data by kfold (%s)" % k)
        kfold = StratifiedKFold(n_splits=k, shuffle=shuffle_data, random_state=random_state)
        train_data = self.train_data
        X_train = self.X_train
        folded_data = kfold.split(X_train, train_data.target)

        for i in range(k):
            fold = next(folded_data)
            X_t = X_train.iloc[fold[0]]
            X_v = train_data.iloc[fold[0]]
            y_t = X_train.iloc[fold[1]]
            y_v = train_data.iloc[fold[1]]

            yield X_t, X_v, y_t, y_v

    def get_test(self):
        if isinstance(self.test_data, pd.DataFrame):
            return self.train_data, self.X_train, self.test_data, self.X_test
        raise Exception("No test data provided!")

    def get_all_text(self):
        return self.df_all['question_text']

    # TODO add
    #       - use DuckTyping.
    #       - statistics retrievers. >>> Most common, uncommon etc. :)


if __name__ == '__main__':
    dr = DataReader('quora/train.csv')
