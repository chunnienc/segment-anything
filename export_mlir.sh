#!/bin/bash

cd $(dirname $0)

python scripts/export_mlir.py \
    --checkpoint "./models/sam_vit_h_4b8939.pth" \
    --model-type "vit_h" \
    --output "./models/sam.mlir"
