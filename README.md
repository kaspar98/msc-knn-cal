# Requirements
Python 3.8.5 and Anaconda package manager was used for the experiments.
To recreate the virtual environment, download  [Anaconda](https://www.anaconda.com/) and run 

`conda env create -n ENVNAME --file environment.yml`

# Data
## Synthetic experiment data
To construct the true calibration map used in the synthetic experiment, the ResNet-110 predictions on CIFAR-5m are used.

Download the logits in pickle format from [here](https://owncloud.ut.ee/owncloud/index.php/s/bJzpBbHmYDoaG3T) \cite{fit-on-the-test}.

Place the downloaded logit files into the folder "logits_5m/logits_resnet_s7".

## Real experiments data
For the real data experiments the predictions of ResNet-110, DenseNet-40, ResNet Wide 32 on CIFAR-10 and CIFAR-100 are used.

Download the logits in pickle format from [here](https://github.com/markus93/NN_calibration) \cite{dirscaling}.

Place the downloaded logit files into the folder "logits".

# Running the experiments
## Real data
To train different post-hoc calibration methods and save their predictions, run `python train_and_save_calibrators.py`.
The predictions will be saved in pickle format to the folder "results".

By default the following model-dataset combinations will be used:
  * densenet40_c10,  
  * densenet40_c100,  
  * resnet_wide32_c10,  
  * resnet_wide32_c100,
  * resnet110_c10,
  * resnet110_c100.
  
To run for only some of the combinations, modify the for-loop in line 241 of "train_and_save_calibrators.py".

By default the following calibration methods will be trained for the model-dataset combinations:
* Dirichlet scaling with ODIR,
* matrix scaling with ODIR
* vector scaling,
* temperature scaling (TS),
* composition of TS and decision calibration with 2 decisions,
* KNN calibration with Kullback-Leibler (KL) divergence,
* composition of TS and KNN calibration with KL divergence,
* composition of TS and KNN calibration with Euclidean distance,
* kernel calibration with Dirichlet kernel,
* composition of TS and kernel calibration with Dirichlet kernel,
* composition of TS and kernel calibration with RBF kernel,
* random calibration forests,
* composition of TS and random calibration forests.

For composition methods, temperature scaling is always applied first.
To run only some of the methods, modify the for-loop in line 250 of "train_and_save_calibrators.py".

**Including IOP**.
To also include the results of diagonal subfamily of intra order-preserving methods, clone the original repository https://github.com/AmirooR/IntraOrderPreservingCalibration.
For each model-dataset combination run their scripts:

`python -u calibrate.py --exp_dir exp_dir/{dataset}/{model}/DIAG --singlefold True`

`python -u evaluate.py --exp_dir exp_dir/{dataset}/{model}/DIAG --save_logits True`

The exact instructions to run the IOP method are provided in their repository.
Copy the saved files "scores.npy" and "logits.npy" for each model-dataset combination into the corresponding folders "results/precomputed/iop/{model}_{dataset}".
To modify the saved logit files to be compatible with the Table generation code, uncomment the line 255 in "train_and_save_calibrators.py" and rerun it for "iop_diag" only (modify the for-loop in line 250).

Finally, to recreate the tables of real data experiments, run the notebook "Tables of real data experiments.ipynb".

## Synthetic data
To have the IOP method available for the synthetic experients, clone the original repository to the the root directory of this repository:

`git clone https://github.com/AmirooR/IntraOrderPreservingCalibration`.

To recreate the figures and the table with results for synthetic experiments, run the notebook "Figures and synthetic experiment.ipynb".
