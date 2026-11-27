# MotionSense 1D CNN external development pilot

Locked 2026-11-26 before model training or validation results. This is an independent dataset replication of the training comparison, not transfer of a UCI classifier.

- Official archive: https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/data/A_DeviceMotion_data.zip. Archive directory metadata may be inspected; held-out subject CSV contents must not be read.
- Training subjects: 1,4,6,7,8,10,11,12,13,15,16,17,20,21,23,24.
- Validation subjects: 2,5,14,19. Sealed test subjects: 3,9,18,22.
- Six local labels in fixed order: downstairs, upstairs, walking, jogging, sitting, standing. These differ from UCI HAR labels.
- Nine native channels in order: userAcceleration.x/y/z, rotationRate.x/y/z, gravity.x/y/z. Do not use row index or attitude as features. These are CoreMotion-derived measurements, not untouched hardware readings. Gravity channels differ from UCI total-acceleration channels; no claim of identical input representation across datasets.
- Use nominal 50 Hz, 128-sample windows, stride 64. Window separately within each subject and trial; discard incomplete trailing windows. No windows across trial boundaries. Do not perform activity- or validation-dependent trimming. The archive has row indices rather than physical timestamps; actual sampling jitter cannot be audited here.
- Standardization: channel mean and standard deviation over training windows and their time steps only, applied unchanged to validation.
- Architecture: Conv1d(9,w,5,padding=2), ReLU, MaxPool1d(2), Conv1d(w,2w,5,padding=2), ReLU, AdaptiveAvgPool1d(1), Linear(2w,6).
- Teacher: w=32, hard CE, seed 20261120, 80 epochs. Freeze the best teacher within those 80 epochs for every student and both student training budgets.
- Students: w=8, seeds 20261121/22/23, three losses CE / uniform label smoothing epsilon 0.1 / KD with temperature 2 and teacher weight 0.5. KD loss = 0.5 hard CE + 0.5 * T^2 * KL(teacher || student), reduction=batchmean. Teacher logits are detached and cached in training-row order.
- Adam lr=0.001, weight_decay=0, batch_size=128, no scheduler/augmentation/dropout/BN. Students train for 160 epochs, with best80/last80/best160/last160 snapshots. Best means max validation macro-F1, then min hard-label NLL, then earliest epoch. All classes are included in macro-F1.
- Match student initial state and full minibatch order within each seed across losses. Compare paired differences. Three seeds are optimization repeats on one person split, not three independent population evaluations.
- No test evaluation. No further hyperparameter search prompted by these results. Report all methods and seeds, including negative results.
- Cost rule: if the teacher estimate after 5 epochs exceeds 600 seconds, stop this external pilot with a documented incomplete status. If it passes, run the fixed nine-student matrix. Training wall times are diagnostic only because another CPU training job may run concurrently.

Question: does the CE/LS/KD ordering and its training-budget dependence recur in an independently collected sensor dataset? This pilot does not establish cross-dataset transfer or a publishable new algorithm.
