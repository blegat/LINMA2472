# Homework 1: GAN (in fact CGAN)

All the instructions for this homework are in the pdf on moodle and the notebook `gan.ipynb`.

The details on how to run the notebook on Google Colab is in the notebook itself.
Alternatively, you can use the CECI clusers by following the instructions below.

## CECI cluster

In order to use the CECI clusters, you need a CECI account.
If you don't already have an account (if you don't know whether you have an account, chances are you don't have one), first [create one](https://login.ceci-hpc.be/init/).
You will receive an email, follow the link in the email and in the field labelled "Email of Supervising Professor", enter `benoit.legat@uclouvain.be`.
Follow the steps detailed [here](https://support.ceci-hpc.be/doc/_contents/QuickStart/ConnectingToTheClusters/index.html) in order to download your private key, create the corresponding public key and create the file `.ssh/config`.

Follow [this guide](https://support.ceci-hpc.be/doc/_contents/ManagingFiles/TransferringFilesEffectively.html) to copy the notebook as well as the three scripts `install.sh`, `load.sh`, `notebook.sh` and `submit.sh` from your computer to the cluster. For instance, with `scp` you can copy the notebook from your computer with:
```sh
(your computer) $ scp gan.ipynb manneback:.
```
and copy the scripts to the cluster with:
```sh
(your computer) $ scp install.sh load.sh notebook.sh submit.sh manneback:.
```

You should now be able to connect to the manneback cluster with
```sh
(your computer) $ ssh manneback
```

Start by installing the dependencies using (you should do this only once, not everytime you connect to the cluster):
```sh
(manneback cluster) $ bash install.sh
```

In order to provide more resources to JupyterLab, [submit the job `bash notebook.sh` with Slurm](https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html).
The file `submit.sh` gives an example of submission script to use to request a GPU (see [here](https://www.ceci-hpc.be/scriptgen.html) for a helper for writing your own submission script). You can use it with
```sh
(manneback cluster) $ sbatch submit.sh
```
The output produced by the job is written in the file `slurm-<JOBID>.out` where `<JOBID>` is the job id listed in the `JOBID` column of the table outputted by
```sh
(manneback cluster) $ squeue --me
```
At the end of the file, you should copy-paste the url given by Jupyter as you will need you give this url (appended with `/24`) to `sshuttle` in the next step.
Now, follow the instructions [here](https://support.ceci-hpc.be/doc/_contents/UsingSoftwareAndLibraries/Jupyter/index.html#connect-to-the-jupyterhub-interface) to use this instance of JupyterLab from a web browser of your computer.

Note that if you do `(manneback cluster) $ bash notebook.sh` directly without using `sbatch` or `srun`, the notebook will run on the *login node* which has limited resources as it is only meant for you to connect and send jobs via Slurm that are executed on *compute nodes*, you will also not have any GPU on the login node.

If you have any issues following these steps, don't hesitate to let us know on the [Class Forum on Moodle](https://moodle.uclouvain.be/mod/forum/view.php?id=43330).
