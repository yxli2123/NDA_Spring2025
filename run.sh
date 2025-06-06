python mlp_dropout.py \
  --data  ../heart_failure_clinical_records_dataset.csv \
  --label_name DEATH_EVENT \
  --test_size 0.25 \
  --batch_size 64 \
  --epochs 10 \
  --lr 0.05 \
  --hidden_dims 6 3 \
  --eval_dropout_masked 0.4 \
  --eval_dropout_original 0.8 \
  --dropout 0.1 \
  --device mps \
  --seed 999 \
  --repeats 100 \
