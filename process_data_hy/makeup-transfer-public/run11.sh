python src/handle_dataset.py --type filter --input data/raw/ --output data/crop/ --detector stasm
python src/handle_dataset.py --type landmark --input data/crop/ --output data/crop/landmark.pk
python src/handle_dataset.py --type blend --input data/crop/ --output data/crop/blend/ --landmark_input data/crop/landmark.pk --keep_eye_mouth --include_forehead --adjust_color --adjust_lighting











