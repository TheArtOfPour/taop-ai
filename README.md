# taop-ai
TF/Keras ML code for https://theartofpour.com

Thank you to everyone who attended my talk at this year's HomebrewCon.

I'll be working over the next few days on getting any remaining code/data deployed, as well as setting licenses and documentation.

Case in point, this README.

Cheers!

*The trained models can be found as .h5 files in the taop-api /models*

- `python 1-sql_to_pickle.py` to query and convert data to pickle format

- `python 2-pickle_to_one_hot.py` to convert data to one-hot format

- `python 3-train_model.py` to train the TF model

- `python test.py` | `python test_individual.py` | `python test_specific.py`
for specific testing runs

@todo: Dynamic input dimensions