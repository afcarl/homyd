from keras.models import Sequential
from keras.layers import Dense

from csxdata.utilities.loader import pull_mnist_data

from homyd import Dataset


def build_homyd_dataset():
    ds = Dataset.from_multidimensional(*pull_mnist_data(split=0, fold=True), dtypes=("float32", None))
    ds.set_embedding("onehot")
    ds.split_new_subset_from(source_subset="learning", new_subset="testing", split_ratio=0.2)
    return ds


def create_ann(inshape, outshape):
    brain = Sequential([
        Dense(60, input_shape=inshape, activation="tanh"),
        Dense(outshape, activation="softmax")
    ])
    brain.compile(optimizer="adam", loss="categorical_crossenropy", metrics=["acc"])
    return brain


def xperiment():
    data = build_homyd_dataset()
    ann = create_ann(*data.shapes)
    ann.fit_generator(data.batch_stream(), steps_per_epoch=len(data) // 32, epochs=30,
                      validation_data=data.table(subset="testing"))


if __name__ == '__main__':
    xperiment()
