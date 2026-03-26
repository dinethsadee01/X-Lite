======================================================================
FULL DATASET TRAINING
======================================================================
Model: efficientnet_b0_performer
Epochs: 50 (early stopping patience=8)
Batch size: 64 | LR: 5e-05
Loss: Weighted BCE with class weights
======================================================================

Device: cuda
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
CUDA Version: 11.8

Preparing data loaders...
Train samples: 109,852
Val samples: 9,073
Loaded 109852 images
Loaded 9073 images
Train batches: 1716
Val batches: 142

======================================================================
TRAINING: efficientnet_b0_performer
======================================================================
Unexpected keys (bn2.num_batches_tracked, bn2.bias, bn2.running_mean, bn2.running_var, bn2.weight, classifier.bias, classifier.weight, conv_head.weight) found while loading pretrained weights. This may be expected if model is being adapted.
Parameters: 4,059,723
Model Size: 15.68 MB
Backbone: efficientnet_b0
Attention: performer

Class distribution (showing bottom 5 and top 5):
  Least common:
    Hernia               219
    Pneumonia            1,402
    Fibrosis             1,650
    Edema                2,270
    Emphysema            2,472
  Most common:
    Nodule               6,197
    Atelectasis          11,326
    Effusion             13,068
    Infiltration         19,503
    No_Finding           59,119
Loss: Focal Loss (α=0.25, γ=2.0)
Checkpoints: C:\Users\User\Sadeepa\X-Lite\ml\models\new checkpoints\efficientnet_b0_performer_full_dataset_15class_patientwise_lol        
Learning Rate: 5e-05
Gradient Clipping: 5.0
Per-epoch checkpoints enabled: C:\Users\User\Sadeepa\X-Lite\ml\models\new checkpoints\efficientnet_b0_performer_full_dataset_15class_patientwise_lol\epoch_checkpoints
======================================================================
STARTING TRAINING
======================================================================
Device: cuda
Model parameters: 4,059,723
Training batches: 1716
Validation batches: 142
Mixed precision: True
======================================================================
Epoch 1 [Train]:   0%|                                                                                          | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 1 [Val]:   0%|                                                                                             | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.6605)
  ✓ Saved epoch checkpoint: model_epoch_001_0.6605.pth

Epoch 1/50 (335.4s)
  Train Loss: 0.0529 | AUC: 0.5503 | F1: 0.0627
  Val Loss:   0.0223 | AUC: 0.6605 | F1: 0.0434
  LR: 0.000050
  🏆 New best AUC: 0.6605
Epoch 2 [Train]:   0%|                                                                                          | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 2 [Val]:   0%|                                                                                             | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.7168)
  ✓ Saved epoch checkpoint: model_epoch_002_0.7168.pth

Epoch 2/50 (335.4s)
  Train Loss: 0.0256 | AUC: 0.5950 | F1: 0.0383
  Val Loss:   0.0204 | AUC: 0.7168 | F1: 0.0394
  LR: 0.000050
  🏆 New best AUC: 0.7168
Epoch 3 [Train]:   0%|                                                                                          | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 3 [Val]:   0%|                                                                                             | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.7480)
  ✓ Saved epoch checkpoint: model_epoch_003_0.7480.pth

Epoch 3/50 (339.5s)
  Train Loss: 0.0233 | AUC: 0.6317 | F1: 0.0350
  Val Loss:   0.0192 | AUC: 0.7480 | F1: 0.0267
  LR: 0.000050
  🏆 New best AUC: 0.7480
Epoch 4 [Train]:   0%|                                                                                          | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 4 [Val]:   0%|                                                                                             | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.7734)
  ✓ Saved epoch checkpoint: model_epoch_004_0.7734.pth

Epoch 4/50 (338.8s)
  Train Loss: 0.0219 | AUC: 0.6610 | F1: 0.0342
  Val Loss:   0.0186 | AUC: 0.7734 | F1: 0.0326
  LR: 0.000050
  🏆 New best AUC: 0.7734
Epoch 5 [Train]:   0%|                                                                                          | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 5 [Val]:   0%|                                                                                             | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.7871)
  ✓ Saved epoch checkpoint: model_epoch_005_0.7871.pth

Epoch 5/50 (339.5s)
  Train Loss: 0.0209 | AUC: 0.6885 | F1: 0.0354
  Val Loss:   0.0183 | AUC: 0.7871 | F1: 0.0408
  LR: 0.000050
  🏆 New best AUC: 0.7871
Epoch 6 [Train]:   0%|                                                                                          | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 6 [Val]:   0%|                                                                                             | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8068)
  ✓ Saved epoch checkpoint: model_epoch_006_0.8068.pth

Epoch 6/50 (348.7s)
  Train Loss: 0.0202 | AUC: 0.7072 | F1: 0.0369
  Val Loss:   0.0177 | AUC: 0.8068 | F1: 0.0403
  LR: 0.000050
  🏆 New best AUC: 0.8068
Epoch 7 [Train]:   0%|                                                                                          | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 7 [Val]:   0%|                                                                                             | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8210)
  ✓ Saved epoch checkpoint: model_epoch_007_0.8210.pth

Epoch 7/50 (351.6s)
  Train Loss: 0.0196 | AUC: 0.7253 | F1: 0.0375
  Val Loss:   0.0174 | AUC: 0.8210 | F1: 0.0370
  LR: 0.000050
  🏆 New best AUC: 0.8210
Epoch 8 [Train]:   0%|                                                                                          | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 8 [Val]:   0%|                                                                                             | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8231)
  ✓ Saved epoch checkpoint: model_epoch_008_0.8231.pth

Epoch 8/50 (350.6s)
  Train Loss: 0.0192 | AUC: 0.7459 | F1: 0.0413
  Val Loss:   0.0173 | AUC: 0.8231 | F1: 0.0417
  LR: 0.000050
  🏆 New best AUC: 0.8231
Epoch 9 [Train]:   0%|                                                                                          | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 9 [Val]:   0%|                                                                                             | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8335)
  ✓ Saved epoch checkpoint: model_epoch_009_0.8335.pth

Epoch 9/50 (334.3s)
  Train Loss: 0.0189 | AUC: 0.7587 | F1: 0.0419
  Val Loss:   0.0169 | AUC: 0.8335 | F1: 0.0414
  LR: 0.000050
  🏆 New best AUC: 0.8335
Epoch 10 [Train]:   0%|                                                                                         | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 10 [Val]:   0%|                                                                                            | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8391)
  ✓ Saved epoch checkpoint: model_epoch_010_0.8391.pth

Epoch 10/50 (327.4s)
  Train Loss: 0.0186 | AUC: 0.7681 | F1: 0.0431
  Val Loss:   0.0168 | AUC: 0.8391 | F1: 0.0484
  LR: 0.000050
  🏆 New best AUC: 0.8391
Epoch 11 [Train]:   0%|                                                                                         | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 11 [Val]:   0%|                                                                                            | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8490)
  ✓ Saved epoch checkpoint: model_epoch_011_0.8490.pth

Epoch 11/50 (329.3s)
  Train Loss: 0.0184 | AUC: 0.7766 | F1: 0.0470
  Val Loss:   0.0166 | AUC: 0.8490 | F1: 0.0518
  LR: 0.000050
  🏆 New best AUC: 0.8490
Epoch 12 [Train]:   0%|                                                                                         | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 12 [Val]:   0%|                                                                                            | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8553)
  ✓ Saved epoch checkpoint: model_epoch_012_0.8553.pth

Epoch 12/50 (329.3s)
  Train Loss: 0.0182 | AUC: 0.7837 | F1: 0.0476
  Val Loss:   0.0164 | AUC: 0.8553 | F1: 0.0480
  LR: 0.000050
  🏆 New best AUC: 0.8553
Epoch 13 [Train]:   0%|                                                                                         | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 13 [Val]:   0%|                                                                                            | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8620)
  ✓ Saved epoch checkpoint: model_epoch_013_0.8620.pth

Epoch 13/50 (329.7s)
  Train Loss: 0.0180 | AUC: 0.7930 | F1: 0.0496
  Val Loss:   0.0163 | AUC: 0.8620 | F1: 0.0468
  LR: 0.000050
  🏆 New best AUC: 0.8620
Epoch 14 [Train]:   0%|                                                                                         | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 14 [Val]:   0%|                                                                                            | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8674)
  ✓ Saved epoch checkpoint: model_epoch_014_0.8674.pth

Epoch 14/50 (347.7s)
  Train Loss: 0.0178 | AUC: 0.8006 | F1: 0.0524
  Val Loss:   0.0161 | AUC: 0.8674 | F1: 0.0565
  LR: 0.000050
  🏆 New best AUC: 0.8674
Epoch 15 [Train]:   0%|                                                                                         | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 15 [Val]:   0%|                                                                                            | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8763)
  ✓ Saved epoch checkpoint: model_epoch_015_0.8763.pth

Epoch 15/50 (344.1s)
  Train Loss: 0.0176 | AUC: 0.8085 | F1: 0.0560
  Val Loss:   0.0157 | AUC: 0.8763 | F1: 0.0542
  LR: 0.000050
  🏆 New best AUC: 0.8763
Epoch 16 [Train]:   0%|                                                                                         | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 16 [Val]:   0%|                                                                                            | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8815)
  ✓ Saved epoch checkpoint: model_epoch_016_0.8815.pth

Epoch 16/50 (344.7s)
  Train Loss: 0.0174 | AUC: 0.8169 | F1: 0.0603
  Val Loss:   0.0155 | AUC: 0.8815 | F1: 0.0597
  LR: 0.000050
  🏆 New best AUC: 0.8815
Epoch 17 [Train]:   0%|                                                                                         | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 17 [Val]:   0%|                                                                                            | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8870)
  ✓ Saved epoch checkpoint: model_epoch_017_0.8870.pth

Epoch 17/50 (324.7s)
  Train Loss: 0.0172 | AUC: 0.8226 | F1: 0.0646
  Val Loss:   0.0153 | AUC: 0.8870 | F1: 0.0610
  LR: 0.000050
  🏆 New best AUC: 0.8870
Epoch 18 [Train]:   0%|                                                                                         | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 18 [Val]:   0%|                                                                                            | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8925)
  ✓ Saved epoch checkpoint: model_epoch_018_0.8925.pth

Epoch 18/50 (324.5s)
  Train Loss: 0.0170 | AUC: 0.8297 | F1: 0.0701
  Val Loss:   0.0150 | AUC: 0.8925 | F1: 0.0752
  LR: 0.000050
  🏆 New best AUC: 0.8925
Epoch 19 [Train]:   0%|                                                                                         | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 19 [Val]:   0%|                                                                                            | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.8990)
  ✓ Saved epoch checkpoint: model_epoch_019_0.8990.pth

Epoch 19/50 (323.2s)
  Train Loss: 0.0168 | AUC: 0.8377 | F1: 0.0757
  Val Loss:   0.0146 | AUC: 0.8990 | F1: 0.0827
  LR: 0.000050
  🏆 New best AUC: 0.8990
Epoch 20 [Train]:   0%|                                                                                         | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 20 [Val]:   0%|                                                                                            | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved best model (AUC: 0.9041)
  ✓ Saved epoch checkpoint: model_epoch_020_0.9041.pth

Epoch 20/50 (324.3s)
  Train Loss: 0.0166 | AUC: 0.8418 | F1: 0.0818
  Val Loss:   0.0145 | AUC: 0.9041 | F1: 0.0685
  LR: 0.000050
  🏆 New best AUC: 0.9041
Epoch 21 [Train]:   0%|                                                                                         | 0/1716 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:144: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
Epoch 21 [Val]:   0%|                                                                                            | 0/142 [00:00<?, ?it/s]C:\Users\User\Sadeepa\X-Lite\ml\training\trainer.py:231: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with autocast():
C:\Users\User\Sadeepa\X-Lite\.venv\Lib\site-packages\numpy\ma\core.py:5938: RuntimeWarning: overflow encountered in cast
  np.copyto(result, result.fill_value, where=newmask)
  ✓ Saved epoch checkpoint: model_epoch_021_0.9022.pth

Epoch 21/50 (324.8s)
  Train Loss: 0.0163 | AUC: 0.8485 | F1: 0.0852
  Val Loss:   0.0143 | AUC: 0.9022 | F1: 0.1062
  LR: 0.000050
Epoch 22 [Train]:   0%|  