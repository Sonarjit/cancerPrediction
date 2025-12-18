from src.text_preprocessing import get_processed_data
from sklearn.model_selection import train_test_split

processed_data = get_processed_data()

y_true = processed_data['Class'].values
processed_data.Gene      = processed_data.Gene.str.replace(r'\s+', '_')
processed_data.Variation = processed_data.Variation.str.replace(r'\s+', '_')


# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]
X_train, test_df, y_train, y_test = train_test_split(processed_data, y_true, stratify=y_true, test_size=0.2)

# split the train data into train and cross validation by maintaining same distribution of output varaible 'y_train' [stratify=y_train]
train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)

def get_train_data():
    return train_df
def get_test_data():
    return test_df
def get_cv_data():
    return cv_df

def get_train_target():
    return y_train
def get_test_target():
    return y_test
def get_cv_target():
    return y_cv