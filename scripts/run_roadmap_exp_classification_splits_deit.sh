PYTHONPATH=${HOME}/projects/made/landmarks-recognition/andrey-research-repo python retrieval/single_experiment_runner.py \
  'experience.experiment_name=GLDv2_ROADMAP_classification_splits_v2_batch${dataset.sampler.kwargs.batch_size}' \
  'experience.log_dir=${env:HOME}/projects/made/landmarks-recognition/andrey-research-repo/experiments/ROADMAP' \
  experience.seed=42 \
  experience.max_iter=1000 \
  optimizer=gldv2_deit \
  model=deit \
  transform=gldv2 \
  dataset=gldv2_10k_classification_splits \
  dataset.sampler.kwargs.batch_size=128 \
  loss=roadmap


#   experience.resume=${HOME}/projects/made/landmarks-recognition/andrey-research-repo/experiments/ROADMAP/GLDv2_ROADMAP_classification_splits128/weights/rolling.ckpt \
