# MS-Diagnostic-Tool
A machine learning tool for non-invasive diagnosis of multiple sclerosis.

Early, non-invasive multiple sclerosis (MS) diagnosis can provide significant cost savings and clinically relevant measures for MS diagnosis and treatment. CLight Technologies (Berkeley, CA) has provided data and a base model for distinguishing MS patients from healthy patients using eye motion detection as a primary input. The aim of the present work (MS Diagnosis) is to perform a hyperparameter analysis of CLight's base model and to improve the model to get it closer to a production ready code set.

Hyperparameter scans in model architecture space are performed on the best base model using SigOpt's bayesian optimization toolkit. Additionally, a constrained scan of model architecture space is performed, enforcing specific architectural shapes that appear most promising from hand-tuning investigations.

# About the Code
Only the model, analysis, and hyperparameter tuning portions of the pipeline are provided. The entire code base from dataset to model instantiation is proprietary and cannot be described in much detail. Broadly described, it is a DNN model that performs a binary classification (MS patient or healthy patient) based upon X-Y positional data collected from several hundred subjects. The data are transformed, truncated, and pre-processed in varous ways to provide meaningful input to the genearlized vector input of the model for each training trial. Data are cross validated by at least 5-folds and pocket-best models are retained for further analysis.

# Search_Params.py
This is the top-level code that provides the entry and exit point for hyperparameter investigation of this model. All relevant features and/or parameters have been obfuscated and any hard-coded values are given dummy values. The Search_Params file constructs an "experiment" loop which trains and cross-validates the model on the full data set using chosen model parameters.

The model is optimized to maximize validation accuracy accross 5 folds. The chosen model parameters are selected with the help of a Bayesian Optimization routine obtained from the SigOpt service. SigOpt has generously provided a free temporary user token for Insight Fellows. Other Bayesian Optimization procedures are available in open source (i.e. Sci-Kit's SKOPT). However, SigOpt's service bolts on top of the pipeline and did not require substantial dataset reconstruction (whereas SKOPT required a batch presentation of input).

# data.py, plt_utils.py, raw_plts.py
These files do simple database creation, plotting, and analysis on data within a hyper-parameter scan results file.

# Running this code
Unfortunately, the code cannot be run without the full model and data set, which cannot be provided to this demonstration folder. All code were executed either on AWS or on CLight's 
