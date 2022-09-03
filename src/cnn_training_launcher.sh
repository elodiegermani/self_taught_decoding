#OAR -O /srv/tempdd/egermani/Logs/job_%jobid%.output
#OAR -E /srv/tempdd/egermani/Logs/job_%jobid%.error

# Parameters
expe_name="data_selection_neurovault"
main_script=$HOME/Documents/transfer_decoding/src/cnn_training.py

echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="/srv/tempdd/egermani/Logs/${CURRENTDATE}_OARID_${OAR_JOB_ID}/"
echo "path log :"
echo $PATHLOG
mkdir $PATHLOG

output_file=$PATHLOG/$OAR_JOB_ID.txt

data_dir=/srv/tempdd/egermani/data_selection_neurovault/data/HCP
out_dir=/srv/tempdd/egermani/transfer_decoding/results
preprocess_type=resampled_masked_normalized
subset=hcp_global_subset_500
model=model_cnn_hcp_task
retrain=kfold
repeatability=1

# The job
# source .bashrc
source /srv/tempdd/egermani/miniconda3/etc/profile.d/conda.sh
source /srv/tempdd/egermani/miniconda3/bin/activate
conda activate workEnv

#conda activate workEnv

# -u : Force les flux de sortie et d'erreur standards à ne pas utiliser de tampon. 
# Cette option n'a pas d'effet sur le flux d'entrée standard
for e in 500
do 
	for b in 64
	do
		for l in 1e-4
		do
			python -u $main_script -d $data_dir -o $out_dir -e $e -b $b -p $preprocess_type -s $subset -m $model -l $l -r $retrain -R $repeatability >> $output_file
done
done
done



