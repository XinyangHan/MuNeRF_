# python src/handle_dataset.py --type filter --input data/raw/ --output data/crop/ --detector stasm
echo "Landmark"
# python src/handle_dataset.py --type landmark --input dataH/crop_try/ --output dataH/crop_try/landmark.pk
echo "Blend"
python src/handle_dataset.py --type blend --input dataH/crop_try/ --output dataH/crop_try/blend/ --landmark_input dataH/crop_try/landmark.pk 