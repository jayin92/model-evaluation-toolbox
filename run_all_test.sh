echo "Running All Test"
echo "Copying Files"
python classify.py
echo "Calculating SSIM"
python ssim.py
echo "Calculating VGG Loss"
python vgg_loss.py
echo "Calculating FID"
python -m pytorch_fid real fake
