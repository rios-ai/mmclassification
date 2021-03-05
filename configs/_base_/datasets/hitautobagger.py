# CURRENTLY CONFIGURED FOR 96x96 INPUT, DOWNSAMPLED TO 32x32 FOR CIFAR
# dataset settings
dataset_type = 'HitAutobagger'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', size=96, padding=8),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Resize', size=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(40, -1)),
    dict(type='CenterCrop', crop_size=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_prefix='/home/anthony/code/sku_classifier/mmclassification/data/train',
        ann_file='/home/anthony/code/sku_classifier/mmclassification/data/train/train_annotations.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/anthony/code/sku_classifier/mmclassification/data/val',
        ann_file='/home/anthony/code/sku_classifier/mmclassification/data/val/val_annotations.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/home/anthony/code/sku_classifier/mmclassification/data/test',
        ann_file='/home/anthony/code/sku_classifier/mmclassification/data/test/test_annotations.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')
