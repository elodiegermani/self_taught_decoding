#OAR -O /srv/tempdd/egermani/Logs/job_%jobid%.output
#OAR -E /srv/tempdd/egermani/Logs/job_%jobid%.error

# Parameters
expe_name="cnn_training"
main_script=$HOME/Documents/self_taught_decoding/src/cnn_training.py

data_dir=$HOME/Documents/self_taught_decoding/data/preprocessed/HCP_dataset
out_dir=$HOME/Documents/self_taught_decoding/data/derived
preprocess_type=resampled_masked_normalized
subset=hcp_dataset_50
model=model_cnn_4layers
retrain=no
classif=contrast
valid=hp

# The job
# source .bashrc
source /srv/tempdd/egermani/miniconda3/etc/profile.d/conda.sh
source /srv/tempdd/egermani/miniconda3/bin/activate
conda activate workEnv

#conda activate workEnv

# -u : Force les flux de sortie et d'erreur standards à ne pas utiliser de tampon. 
# Cette option n'a pas d'effet sur le flux d'entrée standard
for e in 500 200
do 
	for b in 32 64
	do
		for l in 1e-4 1e-5
		do
			python -u $main_script -d $data_dir -o $out_dir -e $e -b $b -p $preprocess_type -s $subset -m $model -l $l -r $retrain -v $valid -c $classif 
done
done
done



