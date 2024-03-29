eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

/fp/homes01/u01/ec-krimhau/.local/bin/papermill 01_train_high_vs_med_low_top50_hp.ipynb 01_OUTPUT_train_high_vs_med_low_top50_hp.ipynb --log-level ERROR

