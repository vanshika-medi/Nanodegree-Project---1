# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about the financial and personal details of customers of a Portugese Bank. Our aim is to determine whether a customer will subscribe to the bank term deposit or not.

We initially define a Logistic Regression model and then tuned the hyperparameters of the model using Azure HyperDrive.

## Scikit-learn Pipeline

#Architecture
We apply a list of tuning parameters or transforms and then a final estimator. The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters

#Data
We initially retrieve the dataset usig TabularDatasetFactory. Cleaning the data is a very important step hence we define a function clean_data(ds). Where ds is the the dataset being passed. We then apply OneHot Encoding, inary classification, Logistic Regression to train the data.

#Hyperparameter Tuning
A SKLearn estimator which is used for training in Scikit-learn experiments is used here and it takes training scripts and performs the training on the compute. This estimator will later be passed to the HyperDrive Config script. Then a HyperDrive Config is created using the estimator, parameter sampler and a policy and the HyperDrive run is executed in the experiment. Then a HyperDrive Config is created using the estimator, parameter sampler and a policy and the HyperDrive run is executed in the experiment.
The hyperparameters which are needed to be tuned are defined in the parameter sampler. The hyperparameters that can be tuned here are C and max_iter. C is the inverse regularization parameter and max_iter is the maximum number of iterations. 
The train.py script contains all the steps needed to train and test the model which are data retrieval, data cleaning and pre-processing, data splitting into train and test data, defining the scikit-learn model and training the model on train data and predicting it on the test data to get the accuracy and then saving the model. Finally ,the best run of the hyperdrive is noted and the best model in the best run is saved.

#Benefits of Parameter Sampler
The parameter sampler is used to provide different choices of hyperparameters to choose from and try during hyperparameter tuning using hyperdrive.
For this project, we use RandomParameterSampling 

#Benefits of Early Stopping Policy
Early Stopping policy in HyperDriveConfig and it is useful in stopping the HyperDrive run if the accuracy of the model is not improving from the best accuracy by a certain defined amount after every given number of iterations. We use Bandit Policy in this project. It saves a lot of computational resources.

## AutoML
AutoML means Automated ML which means it can automate all the process involved in a Machine Learning process. 
Automl makes thing a lot better , easier and faster. Applying AutoML here saves us the time and resources and gives us results in a much efficient manner.
In this project, after applying automl, we test out the following models: Standard Scaling, Min Max Scaling, Sparse Normalizer, MaxAbsScaler, etc.
To run AutoML, one needs to use AutoMLConfig class just like HyperdriveConfig class and need to define an automl_config object and setting various parameters in it which are needed to run the AutoML like training_data , iterations, primary_metrics etc.
  
## Pipeline comparison
HyperDrive Accuracy: 0.91442097596504
AutoML Accuracy: 0.9176 (VotingEnsemble/StackEnsemble)
AutoML has a slighty better accuracy score then HyperDrive. This difference might be because the model of the best AutoML run was a different model than the logistic regression applied in Hyper Drive.

## Future work
For this or future projects, we could probably use GridSampling insetead or Random Sampling as it is much more exhaustive and would give better results.

## Proof of cluster clean up
The snapshot is attached below:
