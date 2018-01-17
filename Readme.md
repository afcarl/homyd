# HoMyD

Python library to Hold My Data for learning algorithms.

Light wrapper around the pandas DataFrame which features subsetting, transforming and
iterating over a dataset with lazy evaluation.

The core concepts of the library:

## LearningTable

The learning table is an (X, Y) tuple of numpy arrays with X holding the learning
data (independent variables) and Y holding the available labels or dependent variables
for fitting.

The *LearningTable* constructor illustrates the components which define a learning table

- **df**: Pandas DataFrame holding the raw data table
- **labels**: columns of **df** which are to be used as labels or dependent variables (Y)
- **paramset**: columns of **df** which are to be used as independent variables (X)

Attributes of *LearningTable*:
- **X**: access the dependent variable matrix as a NumPy NdArray
- **Y**: access the independent variable vector/matrix as a NumPy NdArray
- **raw**: reference to the unmodified dataframe used in the construction
- **labels**: list of column names of **Y**
- **paramset**: list of column names of **X**

Methods of *LearningTable* are applying operations to **X** and **Y** together
- **from_xlsx**: *class method* for reading an xlsx file
- **shapes**: *property* which returns the dimensions of **X** and **Y**
- **dropna**: drops NaN rows based on **labels** and **paramset**
- **reset**: restore the LearningTable to its original form (revert operations)
- **batch_stream**: generator for iterating over (x, y) subsamples
- **split**: splits the dataset according to the supplied argument alpha ratio 
- **copy**: returns a copy of the learning table
- **merge**: concatenates two learning tables rowwise
- **shuffle**: shuffles **X** and **Y** together

**\_\_iter\_\_** and **\_\_len\_\_** are defined on *LearningTable*, so instances can be iterated over
and len() returns the number of samples available in the table. 

## Dataset
The dataset groups together multiple *LearningTable* instances of the same learning data.
It is intended to be used for handling subsets of the same data for i.e. cross validation.
Appliing a transformation (like dimensionality reduction on standardization) to **X** is
normally done by fitting the transformator model to only the learning data and
then transforming any other subset of the data.

The constructor of Dataset takes the following arguments:
- **data**: an instance of *LearningTable*
- **transformation**: an instance of a *Transformation* subclass defined in
*homyd.transformation*. Used to transform the independent variables (**X**)
- **embedding**: an instance of an *EmbeddingBase* subclass defined in *homyd.embedding*.
Used to transform/embed/dummycode the dependent variables (**Y**).
- **dropna**: controls whether to drop NaN values from the source dataset. Must be True
if transformation or embedding is applied.

The constructor automatically assigns the source data to the *"learning"* subset.
Different subsets may be accessed through the [] **\_\_getitem\_\_** operator. 

Methods defined on *Dataset*:
- **add_subset**: add a named subset to the *Dataset*. Excepts a *name* and a
*LearningTable* instance.
- **split_new_subset_from**: given *split_ratio*, splits a new subset named
*new_subset* from *souce_subset*.
- **batch_stream**: yields batches of size *batchsize* from the specified *subset*
- **table**: returns **X** and **Y** numpy arrays of specified *subset*.
