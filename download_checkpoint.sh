#!/bin/bash

cd $(dirname $0)

checkpoint_url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

mkdir -p ./models

wget -nc "$checkpoint_url" -P "./models"