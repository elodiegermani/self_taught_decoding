#OAR -O /srv/tempdd/egermani/Logs/job_%jobid%.output
#OAR -E /srv/tempdd/egermani/Logs/job_%jobid%.error

# Parameters
expe_name="data_selection_neurovault"
main_script=$HOME/Documents/transfer_decoding/src/autoencoder_training.py

echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="/srv/tempdd/egermani/Logs/${CURRENTDATE}_OARID_${OAR_JOB_ID}/"
echo "path log :"
echo $PATHLOG
mkdir $PATHLOG

output_file=$PATHLOG/$OAR_JOB_ID.txt

data_dir=/srv/tempdd/egermani/data_selection_neurovault/data/
out_dir=/srv/tempdd/egermani/transfer_decoding/results
preprocess_type=resampled_masked_normalized
subset=global_set
model=model_cnn
learning_rate=1e-4

# The job
# source .bashrc
source /srv/tempdd/egermani/miniconda3/etc/profile.d/conda.sh
source /srv/tempdd/egermani/miniconda3/bin/activate
conda activate workEnv

#conda activate workEnv

# -u : Force les flux de sortie et d'erreur standards à ne pas utiliser de tampon. 
# Cette option n'a pas d'effet sur le flux d'entrée standard
python -u $main_script -d $data_dir -o $out_dir -e 1000 -b 30 -p $preprocess_type -s $subset -m $model -l $lr >> $output_file


