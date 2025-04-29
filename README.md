# DAT550-Animal-CLEF

This repository is used to train and test ReID-models for the Animal-CLEF task.

## Project Structure

In the model folder are several scripts that serve their individual part in the model training-validation-pipeline.

## Pipeline Overview

The pipeline works like this:

1. `model/run_arg.py` gets launched with the desired parameters, using a shell script from the folder "shell_scripts"
2. `model/run_arg.py` runs a training loop with the given parameters
3. `model/run_arg.py` runs `model/plot_metrics.py` which plots training_loss vs validation_loss
4. Lastly `model/run_arg.py` runs `model/evaluate_open_set.py` which evaluates the trained model, making several plots that can be inspected to further visually evaluate the model
5. Both `model/arg_run.py` and `model/evaluate_open_set.py` calculate metrics and difference in metrics to a reference model if given one, this is all saved under `model_data/all_model_metrics.csv`
6. Models and their respective plots/metrics are saved under `model_data/model_name`. The actual model.pth files are not saved to GitHub as they are too big - these exist only on the UIS server under the user "aleks99"

## Infrastructure

Since this required a lot of computational power, the A100 graphics cards in the UIS-lab were used for this task, using slurm to run the `shell_scripts/trainX.sh`. 
The dataset was downloaded to the servers using `shell_scripts/download.sh`.

## Model Optimization Process

When the scripts were optimized, the following method was used to find the best model:

1. From some models that were trained without this entire pipeline in place, we picked the most promising one, metric-wise, looking at geometric_mean
2. This model was then used as a reference model
3. For every new model made, we changed only one parameter from this model to see if it affected the metrics positively or negatively
4. When a new best model was found, this became the new reference model for the next run
5. The runs consisted of 3-6 models being trained simultaneously


