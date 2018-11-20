"""Prepares training and test data"""
import os
import pickle as pkl
from sklearn.model_selection import train_test_split


def main(filelist):
    pixels = []
    labels = []
    for file in filelist:
        f = open(file, "rb")
        data = pkl.load(f)
        pixels.extend(data["imgs"])
        labels.extend(data["labels"])
        f.close()

    x_train, x_test, y_train, y_test = train_test_split(pixels, labels, test_size=0.30,
                                                        random_state=42)

    # if only one file then inherit all information from
    if len(filelist) == 1:
        base = os.path.basename(filelist[0]).split("_")[1:]
        base = "_".join(base)
        filename = "trainTest_" + base

        f = open(filename, "wb")
        data_dic = {"x_train": x_train, "y_train": y_train,
                    "x_test": x_test, "y_test": y_test}
        pkl.dump(data_dic, f)
        f.close()

        print("Training and testing data saved to: %s" % filename)


if __name__ == '__main__':
    main(["../../data/training_data/mixSamp/mix_26-10_HW5837_noHW5837_box150_side20.pkl"])
