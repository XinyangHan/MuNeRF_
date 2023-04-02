# 2022.10.3 HXY
# 我又来征用这个脚本了，让我想想要用哪几个部分

# python /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicface-parsing.PyTorch-master/test.py

 #############################################################################
#      with photos cropped, we only need first and last operation
 #############################################################################

cd /data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-public
# python src/handle_datasetH.py --type filter --input dataH/raw/ --output dataH/$1/ --detector stasm
echo "Landmark"
python src/handle_datasetH.py --type landmark --input dataH/$1/ --output dataH/$1/landmark.pk

echo "Blend"
python src/handle_datasetH.py --type blend1 --input dataH/$1/ --output dataH/$1/blend/ --landmark_input dataH/$1/landmark.pk --keep_eye_mouth --include_forehead









