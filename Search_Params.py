import os
import numpy as np
import itertools
import tensorflow as tf
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import utils.data_formatutils as dfu
import utils.readinutils as readinutils
from utils.dataset import Dataset
from utils.dataset import DatasetGroup

## Models
from models.conv_mlp import CONV_MLP as conv_mlp
from models.fc_mlp import FC_MLP as fc_mlp
from models.regression import REGRESSION as reg

## Params
from params.conv_mlp_params import params as conv_mlp_params
from params.fc_mlp_params import params as fc_mlp_params
from params.regression_params import params as lr_params

#SIGOPT hyper_parameter Scan
from sigopt import Connection

import utils.write_exp_data as wrt_data

"""--------------------------------------
  Model specification is done here:
--------------------------------------"""
model_type = "fc_mlp"
#model_type = "conv_mlp" # TODO: Not implemented yet
#model_type = "reg" # TODO: Not implemented yet
"""--------------------------------------
--------------------------------------"""

"""--------------------------------------
  Data path specification is done here:
--------------------------------------"""
figpath = os.path.expanduser("~")+'/figs/'
datadir = '/data/envision_working_traces/'
patientfile = './data/patient_stats.csv'
"""--------------------------------------
--------------------------------------"""

## Param file selection
if model_type == "conv_mlp":
    params = conv_mlp_params
elif model_type == "fc_mlp":
    params = fc_mlp_params
elif model_type == "reg":
    params = lr_params
else:
    assert False, ("model_type must be 'conv_mlp', 'fc_mlp', or 'reg'")


# duummy used to obfuscate this summary code report. Hyperparameter ranges are proprietary info.proprietary
dummy = 0
conn = Connection(client_token="dummy_WXGTBCXMJDTQFERYOLTEQSHPHBBIBMQY")
ID = None
# Tell SIGOPT the hyperparameter ranges
so_params = []
so_params.append(dict(name = "HP1", type = 'double', bounds = dict(min = dummy, max = dummy)))
so_params.append(dict(name = "HP2", type = 'double', bounds = dict(min = dummy, max = dummy)))
so_params.append(dict(name = "HP3", type = 'double', bounds = dict(min = dummy, max = dummy)))
so_params.append(dict(name = "HP4", type = 'double', bounds = dict(min = dummy, max = dummy)))
so_params.append(dict(name = "HP5", type = 'double', bounds = dict(min = dummy, max = dummy)))
so_params.append(dict(name = "HP6", type = 'double', bounds = dict(min = dummy, max = dummy)))

#Architecture Parameters
so_params.append(dict(name = "HP_A1", type = 'int', bounds = dict(min = dummy, max = dummy)))
so_params.append(dict(name = "HP_A2", type = 'double', bounds = dict(min = dummy, max = dummy)))
so_params.append(dict(name = "HP_A3", type = 'int', bounds = dict(min = dummy, max = dummy)))
so_params.append(dict(name = "HP_A4", type = 'int', bounds = dict(min = dummy, max = dummy)))

# If statement to resume an experiment, otherwise create it new
# Sets up a SIGOPT bayesian optimization session for scanning hyperparamters
if ID is not None:
    so_experiment = conn.experiments(ID).fetch()
else:
    so_experiment = conn.experiments().create(
        name='fc_mlp_1E',
        parameters = so_params,
        metrics=[dict(name='function_value')],
        parallel_bandwidth=1, 
        # Define an Observation Budget (how many hyperparameter combinations to scan)
        observation_budget=195,
)
print("Created experiment: https://app.sigopt.com/experiment/" + so_experiment.id)

## Data setup
trials = readinutils.readin_traces(datadir, patientfile)
trials = [trial for trial in trials if trial.sub_ms.size>0]
if params.truncate_trials:
    trials = dfu.truncate_trials(trials)
if params.trial_split_multiplier is not None:
    trials = dfu.split_trials(trials, multiplier=params.trial_split_multiplier)

patient_trials = [trial for trial in trials if trial.sub_ms == '1']
control_trials = [trial for trial in trials if trial.sub_ms == '0']

data, labels, stats = dfu.make_mlp_eyetrace_matrix(patient_trials, control_trials, params.feature2)
dataset_group_list = DatasetGroup(data, labels, stats, params)

#Heavy use of dummy variable names for obfuscation of proprietary info
if hasattr(params, "transform1") and params.T1:
    dataset_group_list.TRANSFORM1()
if hasattr(params, "transform2") and params.T2:
    dataset_group_list.TRANSFORM2()
if hasattr(params, "feature1") and params.concatenate_F1:
    dataset_group_list.FEATURE1()
if hasattr(params, "feature2") and params.concatenate_F2:
    dataset_group_list.FEATURE2()
if hasattr(params, "feature3") and params.concatenate_F3:
    dataset_group_list.FEATURE3()
if hasattr(params, "feature4") and params.concatenate_F4:
    dataset_group_list.FEATURE4()

params.data_shape = list(dataset_group_list.get_dataset(0)["train"].data.shape[1:])

print("Training on", dataset_group_list.get_dataset(0)["train"].data.shape[0], "data points.")
print("Validating on", dataset_group_list.get_dataset(0)["val"].data.shape[0], "data points.")

## Training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

first_exp = 1
while so_experiment.progress.observation_count < so_experiment.observation_budget:    
    print("\n\n-------\nStarting experiment", so_experiment.progress.observation_count, \
        " out of ", so_experiment.observation_budget)
    #Create a SIGOPT scan
    suggestion = conn.experiments(so_experiment.id).suggestions().create()
    
    # Set up Scan report file
    if (first_exp):
        file_name = 'output/SP_1E4.csv'
        wrt_data.write_head(file_name, suggestion.assignments)
    
    """--------------------------------------
    Using SIGOPT "suggestion" (above), update hyperparameter values
    Set new architectural params for this experiment.
    ----------------------------------""" 
    setattr(params, 'fc_output', {'HP_A1': suggestion.assignments['HP_A2'], 
                                'HP_A2': suggestion.assignments['HP_A2'],
                                'HP_A3': suggestion.assignments['HP_A3'],
                                'HP_A4': dummy})

    #  A unique constraint on architectural shape is applied
    setattr(params, 'fc_output_channels', params.constrain_layers(params.fc_output))
    print("fc_output_channels IS SET TO", params.fc_output_channels)


    # Continue initializing hyperparameters. 
    for key, val in zip(suggestion.assignments.keys(),suggestion.assignments.values()):
        if key == "feature1":
            val = [val]*(len(params.fc_output_channels))
            print("feature2 ={0}".format(val))
            setattr(params, key, val)
        elif key == "feature3":
            val = [val]*len(params.fc_output_channels)
            print("feature4 ={0}".format(val))
            setattr(params, key, val)
        elif (key == 'HP_A1') or (key == 'HP_A2') or (key == 'HP_A3') or (key == 'HP_A4'):
            pass
        else:
            val = val
            print(key, "IS SET TO", val)
            setattr(params, key, val)

    dataset_group_list.reset_counters()
    dataset_group_list.init_results()

    ## Model selection
    if model_type == "conv_mlp":
        model = conv_mlp(params)
    elif model_type == "fc_mlp":
        model = fc_mlp(params)
    elif model_type == "reg":
        model = reg(params)
    else:
        assert False, ("model_type must be 'conv_mlp', 'fc_mlp', or 'reg'")

    ## Perform Monte Carlo Cross Validation (resample and train multiple times)
    while dataset_group_list.crossvalids_completed < params.num_crossvalidations:
        
        #Initiialze some inter-experiment variables and reporting parameters
        data = dataset_group_list.get_dataset(dataset_group_list.crossvalids_completed)
        print(f"Crossvalidation run {dataset_group_list.crossvalids_completed + 1} / {params.num_crossvalidations}")
        pulls_max_acc = 0 #best validation accuracy
        pulls_max_acc_sens = 0 #sensitivity at best val accuracy
        pulls_max_acc_spec = 0 #specificty at best val accuracy
        sys.stdout.flush()
        
        
        ## New TF Session for each Draw
        with tf.Session(config=config, graph=model.graph) as sess:
            ## Need to provide shape if batch_size is used in graph
            sess.run(model.init_op,
                feed_dict={model.x:np.zeros([params.batch_size]+params.data_shape,
                dtype=np.float32)})
            model.write_graph(sess.graph_def)
            sess.graph.finalize() # Graph is read-only after this statement
            epoch_results ={'epoch':[], 'batch_loss':[], 'train_accuracy': [], \
                'val_accuracy':[],'val_sensitivity':[],'val_specificity':[]  }
            while data["train"].epochs_completed < params.num_epochs:
                data_batch = data["train"].next_batch(model.batch_size)
                feed_dict = {model.x:data_batch[0], model.y:data_batch[1]}
                
                 ## Update weights
                sess.run(model.apply_grads, feed_dict)

                if (data["train"].epochs_completed % params.val_frequency == 0
                    and data["train"].batches_this_epoch==1):

                    global_step = sess.run(model.global_step)
                    current_loss = sess.run(model.total_loss, feed_dict)

                    weight_cp_filename, full_cp_filename = model.write_checkpoint(sess)

                    with tf.Session(graph=model.graph) as tmp_sess:
                        val_feed_dict = {model.x:data["val"].data, model.y:data["val"].labels}
                        tmp_sess.run(model.init_op, val_feed_dict)
                        cp_load_file = tf.train.latest_checkpoint(model.cp_save_dir,
                            model.cp_latest_filename+"_weights")
                        model.load_weights(tmp_sess, cp_load_file)
                        run_list = [model.merged_summaries, model.accuracy, model.sensitivity, model.specificity]
                        summaries, val_accuracy, val_sensitivity, val_specificity = tmp_sess.run(run_list,
                            val_feed_dict)
                        model.writer.add_summary(summaries, global_step)

                    """--------------------------------------
                    pulls_max_(...) are Marginal note on best results accrossed epochs and experiments. 
                    Not really a pocket best since these vals are not avveraged accross cross-validation folds. 
                    Does give indication of promising models to be explored.
                    --------------------------------------"""
                    if(pulls_max_acc < val_accuracy):
                        pulls_max_acc = val_accuracy
                        pulls_max_acc_sens = val_sensitivity
                        pulls_max_acc_spec = val_specificity
                        

                    with tf.Session(graph=model.graph) as tmp_sess:
                        tr_feed_dict = {model.x:data["train"].data, model.y:data["train"].labels}
                        tmp_sess.run(model.init_op, tr_feed_dict)
                        cp_load_file = tf.train.latest_checkpoint(model.cp_save_dir,
                            model.cp_latest_filename+"_weights")
                        model.weight_saver.restore(tmp_sess, cp_load_file)
                        train_accuracy = tmp_sess.run(model.accuracy, tr_feed_dict)

                    """--------------------------------------
                    Print/report results/statistics for an experiment indexed by epoch
                    Useful fordynamic allocation of resources accross many experiments
                    --------------------------------------"""
                    num_decimals = 5
                    print("epoch:", str(data["train"].epochs_completed).zfill(3),
                        "\tbatch loss:", np.round(current_loss, decimals=num_decimals),
                        "\ttrain accuracy:", np.round(train_accuracy, decimals=num_decimals),
                        "\tval accuracy:", np.round(val_accuracy, decimals=num_decimals),
                        "\tval sensitivity:", np.round(val_sensitivity, decimals=num_decimals),
                        "\tval specificity:", np.round(val_specificity, decimals=num_decimals))
                    epoch_results.update({'epoch': epoch_results['epoch'] + [data["train"].epochs_completed]})
                    epoch_results.update({'batch_loss': epoch_results['epoch'] + \
                                          [np.round(current_loss, decimals=num_decimals)]})
                    epoch_results.update({'train_accuracy': epoch_results['train_accuracy'] + \
                                          [np.round(train_accuracy, decimals=num_decimals)]})
                    epoch_results.update({'val_accuracy': epoch_results['val_accuracy'] + \
                                          [np.round(val_accuracy, decimals=num_decimals)]})
                    epoch_results.update({'val_sensitivity': epoch_results['val_sensitivity'] + \
                                          [np.round(val_sensitivity, decimals=num_decimals)]})
                    epoch_results.update({'val_specificity': epoch_results['val_specificity'] + \
                                          [np.round(val_specificity, decimals=num_decimals)]})
                    sys.stdout.flush()

            ## Report results for this pull so we can calculate mean and sd for cross validation
            dataset_group_list.record_results(train_accuracy, val_accuracy, val_sensitivity, val_specificity,
                                              pulls_max_acc, pulls_max_acc_sens, pulls_max_acc_spec, 
                                              dataset_group_list.crossvalids_completed, epoch_results)


    ## Report Cross Validated Result
    # result is array trainacc,valacc,sens,spec, maxtracc, sensatmaxtracc, specatmaxtracc
    xvald_means = dataset_group_list.mean_results()
    xvald_sds = dataset_group_list.sd_results()
    print("Cross Validation Completed!",
        "\nMean Final Train accuracy:", np.round(xvald_means[0], decimals=num_decimals), "(SD:",np.round(xvald_sds[0],decimals=num_decimals),")",
        "\nMean Final Val accuracy:", np.round(xvald_means[1], decimals=num_decimals), "(SD:",np.round(xvald_sds[1],decimals=num_decimals),")",
        "\nMean Final sensitivity:", np.round(xvald_means[2], decimals=num_decimals), "(SD:",np.round(xvald_sds[2],decimals=num_decimals),")",
        "\nMean Final specificity:", np.round(xvald_means[3], decimals=num_decimals), "(SD:",np.round(xvald_sds[3],decimals=num_decimals),")",
        "\nMean Best Val accuracy:", np.round(xvald_means[4], decimals=num_decimals), "(SD:",np.round(xvald_sds[4],decimals=num_decimals),")",
        "\nMean Best sensitivity at Best Val acc:", np.round(xvald_means[5], decimals=num_decimals), "(SD:",np.round(xvald_sds[5],decimals=num_decimals),")",
        "\nMean Best specificity at Best Val acc:", np.round(xvald_means[6], decimals=num_decimals), "(SD:",np.round(xvald_sds[6],decimals=num_decimals),")")
 
    wrt_data.write_line(file_name, suggestion.assignments, xvald_means)

    # Update the SIGOPT target value (in this case, xvald_means[4] is mean best accuracy under cross validation)
    value = xvald_means[4]
    
    # Update SIGOPT
    conn.experiments(so_experiment.id).observations().create(
        suggestion=suggestion.id,
        value=value,
        )
    
    # Grab a new SIGOPT suggestion (Bayesian optimization)
    so_experiment = conn.experiments(so_experiment.id).fetch()
    
    sys.stdout.flush()





