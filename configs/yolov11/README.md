# YOLOv11

This directory contains configuration files and metadata for training and
validating YOLOv11 models in MMYOLO. The architecture extends the YOLOv8
baseline with the newly introduced `CIB` blocks for both the backbone and the
neck, and an optionally depthwise-separable detection head.

| Model | Dataset | Metrics | Weight |
| :---- | :------ | :------ | :----- |
| `yolov11_s_syncbn_fast_8xb16-500e_coco` | COCO | box AP **TBD** | _TBD_ |

> **Note:** Pretrained weights are not yet available. You can train the model
> from scratch following the configuration.
