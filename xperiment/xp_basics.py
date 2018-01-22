from keras.models import Sequential
from keras.layers import Dense

from csxdata.utilities.loader import pull_mnist_data

from homyd import Dataset


def build_homyd_dataset():
    ds = Dataset.from_multidimensional(*pull_mnist_data(split=0, fold=True))
    ds.set_embedding("onehot")
    ds.split_new_subset_from(source_subset="learning", new_subset="testing", split_ratio=0.2)
    return ds


def create_ann(inshape, outshape):
    brain = Sequential([
        Dense(60, input_shape=inshape, activation="tanh"),
        Dense(outshape[0], activation="softmax")
    ])
    brain.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return brain


def xperiment():
    batch_size = 32
    data = build_homyd_dataset()
    N = data.subset_sizes["learning"]
    ann = create_ann(*data.shapes)
    ann.fit_generator(data.batch_stream(batch_size), steps_per_epoch=N // batch_size,
                      epochs=30, validation_data=data.table(subset="testing"))


if __name__ == '__main__':
    xperiment()
