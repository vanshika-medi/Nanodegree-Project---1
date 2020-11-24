# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about the financial and personal details of customers of a Portugese Bank. Our aim is to determine whether a customer will subscribe to the bank term deposit or not.

We initially define a Logistic Regression model and then tuned the hyperparameters of the model using Azure HyperDrive.

## Scikit-learn Pipeline

### Architecture
We apply a list of tuning parameters or transforms and then a final estimator. The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters
For this project, we did the following:
* Load data through TabularDatasetFactory feature.
* Cleaned the data using clean_data(ds) function in the train.py file.
* Split the data into training and testing dataset.
* Fit the Scikit-Learn Logistic Regression model and calculate the accuracy.
* Save the model in the folder to be used for the udacity-project.ipynb file where we do Hyperparameter tuning.
* There are two hyperparameters for the logistic regression model: the inverse of regularization strength (C) and max iteration number (max_iter). The aim is to tune               those hyperparameters using HyperDrive.
* We tune the hyperparameters and submit the run to save the best model with the best possible accuracy.

### Data
We initially retrieve the dataset usig TabularDatasetFactory. Cleaning the data is a very important step hence we define a function clean_data(ds). Where ds is the the dataset   being passed. We then apply OneHot Encoding, binary classification, Logistic Regression to train the data.

### Hyperparameter Tuning
Parameters which define the model architecture are known as jhyperparameters and hence to find the best model for our data, we have to tune them. The following parameters are the ones we used in this project:
* **Parameter Sampler** : *RandomParameterSampling* - It defines random sampling across the hyperparameter search space for inverse of regularization strength (C) and maximum                              iteration number (max_iter). It is not as exhaustive as Grid Sampling and hence lacks bias.
* **Early Termination Policy** : *BanditPolicy* - This defines a policy or a set of rules to be followed in case the run gets preempted. We define a slack_factor,                                                 evaluation_interval and delay_evaluation. Any run that does ot fall within the specified slack factor (or slack amount) of the evaluation                                         metric with respect to the best performing run will be terminated. It also helps in avoiding overfitting and hence we choose this policy.
* **Estimator** : An estimator needs to be defined with some sample hyperparameters. The SKLearn estimator for Scikit-Learn model training requires us to input the arguments like the source directory of the file, the name of the training file as well as the compute target being used.
* **HyperDriveConfig** : The HyperDriveConfig is where all the parameters for hyperdrive are set. It includes the above mentioned parameter sampler, early termination policy, estimator along with primary metrics being used, total_runs and max_concurrent_runs. We then submit this hyperdrive_config, retrieve the best possible model and register it.


### Benefits of Parameter Sampler
The parameter sampler is used to provide different choices of hyperparameters to choose from and try during hyperparameter tuning using hyperdrive.
For this project, we use RandomParameterSampling which lacks bias and is overall faster as compared to Grid Sampling. Hence it makes a perfect fit for our project.

### Benefits of Early Stopping Policy
Early Stopping policy in HyperDriveConfig and it is useful in stopping the HyperDrive run if the accuracy of the model is not improving from the best accuracy by a certain defined amount after every given number of iterations. We use Bandit Policy in this project. It saves a lot of computational resources.
Bandit Policy avoids overfitting. For more aggressive savings, we chose the Bandit Policy with a small allowable slack.

## AutoML
AutoML means Automated ML which means it can automate all the process involved in a Machine Learning process. 
Automl makes thing a lot better , easier and faster. Applying AutoML here saves us the time and resources and gives us results in a much efficient manner.
For the Scikit-Learn model, we define the AutoML configurations using the AutoMLConfig class. For our project, we used thr following parameters:
* Load the data in the notebook using the TabularDatasetFactory.
* Clean and split the data into training dataset and testing dataset.
* Get data into TabularDatset form as AutoML does not work on pandas.
* Inititate the AutoMLConfig class which contains parameters like experiment_timeout_minutes (time duration for the experiment to run), task (regression/classification, in this   case classification), label_column_name (target), training data, validationdata, compute target and primary_metric (in this case accuracy).
* We then submit this automl run and after 30 mins get the models applied with the accuracy and finally yhr best run which is the saved.

## Pipeline comparison
HyperDrive Accuracy: 0.91442097596504
AutoML Accuracy: 0.9176 (VotingEnsemble/StackEnsemble)
AutoML has a slighty better accuracy score then HyperDrive. This difference might be because the model of the best AutoML run was a different model than the logistic regression applied in Hyper Drive.
HyperDrive also requires us to specify a lot of parameters hence taking up more time. If we don't specify the model to be applied, HyperDrive will ot be able to run whereas AutoML saves us that much time and work and provides a better accuracy.

## Future work
For this or future projects, we could probably use GridSampling insetead or Random Sampling as it is much more exhaustive and would give better results.

## Proof of cluster clean up
The snapshot is added to the repository.
