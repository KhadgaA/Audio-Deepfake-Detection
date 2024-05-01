# Deep Fake Audio Detection

## Task1: Evaluate model on Custom Dataset

* To evaluate the DF model on the custom dataset, please go to `SSL_Anti-spoofing` folder. Then run `Evaluate_Speech_A3.py` file with arguments `--data_dir` set to dataset path.
* Dataset folder structure:

  > Dataset
  > | __ Real
  > | __ Fake
  >

  **EER Score:** 30.69
  **AUC Score:** 0.2387

## Task2: Model testing on for-2seconds dataset

Follow same process as in task 1. Change the `--data_dir` to the testing set of **`for-2seconds`** dataset.

* Dataset folder structure:

  > Dataset
  > | __ real
  > | __ fake
  >
* Total samples : 1088

**EER Score:** 33.27
**AUC Score:** 0.2456
