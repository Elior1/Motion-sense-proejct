import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

root = "C:/Users/elior/PycharmProjects/data_wrangling/"
def get_ds_infos():
    """
    Read the file includes data subject information.

    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height
    3: age [years]
    4: gender [0:Female, 1:Male]

    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes
    """

    dss = pd.read_csv(root + "data_subjects_info.csv")
    print("[INFO] -- Data subjects' information is imported.")

    return dss


def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.

    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration]

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t + ".x", t + ".y", t + ".z"])
        else:
            dt_list.append([t + ".roll", t + ".pitch", t + ".yaw"])

    return dt_list


def creat_time_series(dt_list, act_labels, trial_codes, mode="mag", labeled=True):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.

    Returns:
        It returns a time-series of sensor data.

    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list * 3)

    if labeled:
        dataset = np.zeros((0, num_data_cols + 7))  # "7" --> [act, code, weight, height, age, gender, trial]
    else:
        dataset = np.zeros((0, num_data_cols))

    ds_list = get_ds_infos()

    print("[INFO] -- Creating Time-Series")
    for sub_id in ds_list["code"]:
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                fname = root + 'A_DeviceMotion_data/' + act + '_' + str(trial) + '/sub_' + str(int(sub_id)) + '.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                vals = np.zeros((len(raw_data), num_data_cols))
                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:, x_id] = (raw_data[axes] ** 2).sum(axis=1) ** 0.5
                    else:
                        vals[:, x_id * 3:(x_id + 1) * 3] = raw_data[axes].values
                    vals = vals[:, :num_data_cols]
                if labeled:
                    lbls = np.array([[act_id,
                                      sub_id - 1,
                                      ds_list["weight"][sub_id - 1],
                                      ds_list["height"][sub_id - 1],
                                      ds_list["age"][sub_id - 1],
                                      ds_list["gender"][sub_id - 1],

                                      trial
                                      ]] * len(raw_data))
                    vals = np.concatenate((vals, lbls), axis=1)
                dataset = np.append(dataset, vals, axis=0)
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]

    if labeled:
        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]

    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset


# ________________________________


ACT_LABELS = ["dws", "ups", "wlk", "jog", "std", "sit"]
TRIAL_CODES = {
    ACT_LABELS[0]: [1, 2, 11],
    ACT_LABELS[1]: [3, 4, 12],
    ACT_LABELS[2]: [7, 8, 15],
    ACT_LABELS[3]: [9, 16],
    ACT_LABELS[4]: [6, 14],
    ACT_LABELS[5]: [5, 13]
}

## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
sdt = ["attitude", "gravity", "rotationRate", "userAcceleration"]
print("[INFO] -- Selected sensor data types: " + str(sdt))
act_labels = ACT_LABELS
print("[INFO] -- Selected activites: " + str(act_labels))
trial_codes = [TRIAL_CODES[act] for act in act_labels]
dt_list = set_data_types(sdt)
dataset = creat_time_series(dt_list, act_labels, trial_codes, mode="mag", labeled=True)
print("[INFO] -- Shape of time-Series dataset:" + str(dataset.shape))

dataset.head()
dataset.describe()['age']
dataset.corr()
dataset.groupby(['age']).mean()['userAcceleration']
dataset.groupby(['trial', 'id']).count().groupby('trial').max()['age']
dataset.groupby(['act']).count()['age'] # Count of each label


y_test = pd.DataFrame()
x_test = pd.DataFrame()
y_train = pd.DataFrame()
x_train = pd.DataFrame()

for j in set(dataset['act']):
    for i in set(dataset[dataset['act'] == j]['trial']):
        if i > 10:
            y_test = pd.concat([y_test, dataset[dataset['act'] == j][dataset[dataset['act'] ==
                                                                             j]['trial'] == i]['act']])
            x_test = pd.concat([x_test, dataset[dataset['act'] == j][dataset[dataset['act'] == j]['trial'] ==
                                         i][dataset.columns[dataset.columns != 'act']]])
        else:
            y_train = pd.concat([y_train, dataset[dataset['act'] == j][dataset[dataset['act'] ==
                                                                              j]['trial'] == i]['act']])
            x_train = pd.concat([x_train, dataset[dataset['act'] == j][dataset[dataset['act'] == j]['trial'] ==
                                              i][dataset.columns[dataset.columns != 'act']]])


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
np.random.seed(123) # for reproducibility
batch_size = 50000

#_x_train = np.reshape(np.array(x_train), (x_train.shape[0], 1, x_train.shape[1]))
#x_test = np.reshape(np.array(x_test), (x_test.shape[0], 1, x_test.shape[1]))
model = keras.models.Sequential()
model.add(keras.layers.Embedding(1081446, 10))
model.add(keras.layers.LSTM(128, dropout=0.7))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(6))
model.add(keras.layers.Activation('softmax'))
# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train
model.fit(x_train, y_train, batch_size=batch_size, epochs=5, verbose=0,
          validation_data=(x_test, y_test))
#----output----


# Evaluate
train_score, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
test_score, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Train score:', train_score)
print('Train accuracy:', train_acc)
print('Test score:', test_score)
print('Test accuracy:', test_acc)


# Random Forest

clf = RandomForestClassifier(n_estimators=1000, max_depth=10,
                             random_state=0)
clf.fit(x_train, np.array(y_train).ravel())
y_pred = clf.predict(x_test)

print(confusion_matrix(y_test, y_pred))
