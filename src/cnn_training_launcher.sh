#OAR -O /srv/tempdd/egermani/Logs/job_%jobid%.output
#OAR -E /srv/tempdd/egermani/Logs/job_%jobid%.error

output_file=$PATHLOG/$OAR_JOB_ID.txt

# Parameters
expe_name="cnn_training"
main_script=/srv/tempdd/egermani/self_taught_decoding/src/cnn_training.py

echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="/srv/tempdd/egermani/Logs/${CURRENTDATE}_OARID_${OAR_JOB_ID}/"
echo "path log :"
echo $PATHLOG
mkdir $PATHLOG

output_file=$PATHLOG/$OAR_JOB_ID.txt

data_dir=/srv/tempdd/egermani/self_taught_decoding/data/preprocessed/BrainPedia_dataset
out_dir=/srv/tempdd/egermani/self_taught_decoding/data/derived/BrainPedia_dataset
preprocess_type=resampled_masked_normalized
retrain=all
classif=tags
valid=perf
l=1e-4

# The job
# source .bashrc
source /srv/tempdd/egermani/miniconda3/etc/profile.d/conda.sh
source /srv/tempdd/egermani/miniconda3/bin/activate
conda activate workEnv

#conda activate workEnv

# -u : Force les flux de sortie et d'erreur standards à ne pas utiliser de tampon. 
# Cette option n'a pas d'effet sur le flux d'entrée standard
for m in model_cnn_5layers
do
	for e in 200
	do 
		for b in 64
		do
			for subset in small_bp_dataset bp_dataset
			do
				python -u $main_script -d $data_dir -o $out_dir -e $e -b $b -p $preprocess_type -s $subset -m $m -l $l -r $retrain -v $valid -c $classif >> $output_file 
done
done
done
done



