#!/bin/bash
# A very simple script to download data and stuffs from Kaggle
# Author: Jimut Bahan Pal, 20-6-2020

touch done_downloading.txt

echo "[1] Downloading from Kaggle ...">> done_downloading.txt

# get the token here real quick
wget "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/9988/868324/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1595509005&Signature=ANVVCs1%2FxsHfSfOv4pZ0EKwqp8kZFBi9mujeSN91y%2BizsUBuVXNdD8QImXWvVOAPgAd8sMhzp0C%2Fg7mH%2FY47v4bYasd%2BNA8EgI2%2F3ng9WxDCKLjNQfPQY53UdSY%2Fm6Bu%2Fy5Iona%2Bxddyhl4u3dxLnI8KSsSZfCgOjwuHJI6zMecOq%2FPrfyzbqb7pGqHa703sOWzouwBptU5qowWsGoirVfr9pDiKJdqzeHIMeQbKkUEc%2F46aGCJA00MvFXu4cP%2FP0OcTGbrA0A9yYmLTnsLDIf2EbR405skweuq%2BqwSqCacm31JUuNHipogoZU%2BBw01sbN%2BtwsdaG17atdwseTswUg%3D%3D&response-content-disposition=attachment%3B+filename%3Dairbus-ship-detection.zip" -O airbus-ship-detection.zip

echo "[2] Downloaded...">> done_downloading.txt
echo "[3] Unzipping...">> done_downloading.txt

unzip  -qq airbus-ship-detection.zip
echo "[4] Finished Unzipping... cleaning!">> done_downloading.txt

rm -rf airbus-ship-detection.zip
echo "[5] Finished Jobs and cleaning...">> done_downloading.txt

