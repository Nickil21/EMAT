#!/bin/bash

#SBATCH --job-name emat-hq-finetune
#SBATCH --account NLP-CDT-SL2-GPU
#SBATCH --partition ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output="/home/%u/logs/%j_%x.out"
#SBATCH --error="/home/%u/logs/%j_%x.err"
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=nickil.maveli@ed.ac.uk

module load miniconda/3
nvidia-smi

# DST_DIR="EMAT" # change to your project root
# cd ${DST_DIR}

# get conda working
source ~/.bashrc

# # create a conda environment
# conda create -n emat -y python=3.8 && conda activate emat
# # install pytorch
# pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html  # GPU
# # install transformers
# pip install transformers==4.14.1
# # install faiss
# pip install faiss-gpu==1.7.1.post3  # GPU
# # install dependencies
# pip install -r requirements.txt

conda activate emat
echo "Environment Activated!"

set -e
set -u


# DST_DIR="EMAT" # change to your project root
# cd ${DST_DIR}

CKPT_DIR="EMAT-ckpt/FKSV-HQ/best_ckpt" # load pre-trained model
EXP_NAME="eval" # set experiment name
DATA_NAME="hq" # datasets: ["nq", "tq", "wq"]

DEVICE="0"

# Train nq-EMAT-FKSV
# use --kvm_fp16 if GPU OOM

CUDA_VISIBLE_DEVICES=${DEVICE} python qa_main.py \
  --project_name="${DATA_NAME^^}-CAT" \
  --exp_name=${EXP_NAME} \
  --query_batch_size=64 \
  --build_mem_batch_size=6000 \
  --batch_local_positive_num=5 \
  --pos_from_top=128 \
  --do_eval \
  --kvm_seg_n=2 \
  --values_with_order \
  --value_layer=7 \
  --value_fusion_method="cat_k_delay+v" \
  --num_values=10 \
  --qa_data_name=${DATA_NAME} \
  --model_name_or_path=${CKPT_DIR} \
  --source_prefix="question: " \
  --per_device_train_batch_size=16 \
  --per_device_eval_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --num_train_epochs=10 \
  --lr_scheduler_type="linear" \
  --num_warmup_steps=1000 \
  --output_dir="./outputs/hq_checkpoints/${EXP_NAME}" \
  --prefix_length=2 \
  --d_key=1536 \
  --key_layer=3 \
  --key_encoder_type="conv" \
  --select_positive_strategy="softmax_sample" \
  --faiss_efsearch=128 \
  --gen_weight=1 \
  --match_weight=1 \
  --key_reduce_method="avg" \
  --qas_to_retrieve_from="PAQ_L1" \
  --local_size=384 \
  --update_kv_embeds \
  --update_local_target_each_batch \
  --update_local_qas \
  --separate_task \
  --value_ae_target="ans" \
  --key_ae_target="question" \
  --repaq_supervision_epoch=-1 \
  --early_stop_patience=4 \
  --negatives_num_each_example=32 \
  --do_test
