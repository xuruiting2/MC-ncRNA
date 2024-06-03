## MM-ncRNAFR: 
#### Multi-modal Contrastive Learning for ncRNA Family Prediction

In this package, we provide the following resources: source codes of the MM-ncRNAFP, pre-trained model weights, and fine-tuning codes. The essential Python packages are listed in the `requirements.txt` file. Additionally, you need to run `pip install -e .` in the main folder. After downloading the [pre-trained model weights](https://drive.google.com/drive/folders/1rfM-aetoUutYp14efdtZdvliGnbW7zuy?usp=drive_link), place the folder in the main directory.

Place the datasets used for fine-tuning the model in the `main/sample_data/ft/rnafamily` folder. You can download the [datasets](https://drive.google.com/drive/folders/1ZVmwO_3ktvRocK739AGOijflsEgIbPtf?usp=drive_link) from this link.


### Fine-tune with pre-trained model
cd main
```
export ROOT_PATH=/no-codingRNA-pretrain/main
export KMER=6
export TASK=rnafamily
export MODEL_PATH=$ROOT_PATH/rna_premodel_0
export DATA_PATH=$ROOT_PATH/sample_data/ft/$TASK
export OUTPUT_PATH_BERT=$ROOT_PATH/task_out/$TASK/bert
export RESULT_OUTDIR=$ROOT_PATH/task_out/$TASK
export OUTPUT_PATH_GCN=$ROOT_PATH/task_out/$TASK/gcn
export InfoGraph_MODEL=$ROOT_PATH/graph_fea/trained_model.pth

python run_fintune.py \
    --model_type rna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name rnafamily \
    --do_train \
    --evaluate_during_training \
    --logging_steps 1000 \
    --data_dir $DATA_PATH \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=5  \
    --per_gpu_train_batch_size=5  \
    --per_gpu_pred_batch_size=5 \
    --num_train_epochs 100 \
    --output_dir $OUTPUT_PATH_BERT \
    --warmup_percent 0.000002 \
    --hidden_dropout_prob 0.01 \
    --overwrite_output \
    --weight_decay 1e-4 \
    --n_process 1 \
    --infoGraph-model $InfoGraph_MODEL \
    --result_dir $RESULT_OUTDIR \
    --gcn-output-path $OUTPUT_PATH_GCN \
    --learning_rate 0.000002 \
```




