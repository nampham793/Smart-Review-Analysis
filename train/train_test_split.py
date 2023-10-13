import json
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

class Split():
    '''
    INPUT: 
        Data path, 
        n_split = (
            2: 50% - 50% (Train - test),
            3: 60% - 40%,
            5: 80% - 20%
        )

    OUTPUT: 80-20 TRAIN-DEV-TEST
        X_train, y_train, X_test, y_test
    '''

    def __init__(self, config_path):
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        self.data_path = config['DATA_PATH']
        self.save_dir_train = config['SAVE_DIR_TRAIN']
        self.save_dir_test = config['SAVE_DIR_TEST']
        self.n_splits = config['N_SPLITS']

    def preprocess_data(self):
        data = pd.read_csv(self.data_path)
        labels = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']
        X = data.drop(columns=labels)
        y = data[labels]

        return X, y

    def count_label_distribution(self, y):
        return y.apply(lambda col: col.value_counts().fillna(0).astype(int))

    def split_data(self, X, y):
        mskf = MultilabelStratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=0)
        train_indices, test_indices = next(mskf.split(X, y))

        train_indices = train_indices.astype(int)  # Convert to NumPy int array
        test_indices = test_indices.astype(int)    # Convert to NumPy int array

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        return X_train, y_train, X_test, y_test

    def save_data_to_csv(self, data, save_path):
        data.to_csv(save_path, index=False)

    def run(self):
        X, y = self.preprocess_data()
        X_train, y_train, X_test, y_test = self.split_data(X, y)

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        self.save_data_to_csv(train_df, self.save_dir_train)
        self.save_data_to_csv(test_df, self.save_dir_test)

if __name__ =="__main__":
    CONFIG_PATH = 'train/split_config.json'
    splitter = Split(config_path=CONFIG_PATH)
    splitter.run()

    print("Completed")



