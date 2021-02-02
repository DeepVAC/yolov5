#!/bin/bash
# COCO 2017 dataset http://cocodataset.org
# Download command: bash data/get_coco.sh
# Default dataset location is next to Yolov5-1:
# /data
#   |
#   --coco
#      |
#      --images
#      |   |
#      |   -- train2017
#      |   |
#      |   -- val2017
#      |   |
#      |   -- test2017
#      |
#      -- instances_train2017.json
#      |
#      -- instances_val2017.json

# Download/unzip labels
d='data/coco' # unzip directory
mkdir $d
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
f='coco2017labels.zip'                                                                 # 68 MB
echo 'Downloading' $url$f ' ...' && curl -L $url$f -o $f && unzip -q $f -d $d && rm $f # download, unzip, remove

# Download/unzip images
d='data/coco/images' # unzip directory
mkdir $d
url=http://images.cocodataset.org/zips/
f1='train2017.zip' # 19G, 118k images
f2='val2017.zip'   # 1G, 5k images
f3='test2017.zip'  # 7G, 41k images (optional)
for f in $f1 $f2; do
  echo 'Downloading' $url$f '...' && curl -L $url$f -o $f # download, (unzip, remove in background)
  unzip -q $f -d $d && rm $f &
done
wait # finish background tasks
