# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/anthony/code/sku_classifier/mmclassification/work_dirs/resnet50_b16x8_cifar10_20200823-882aa7b1.pth'
resume_from = None
workflow = [('train', 1)]
