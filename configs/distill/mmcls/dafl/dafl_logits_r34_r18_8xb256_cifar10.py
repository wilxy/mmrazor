_base_ = [
    'mmcls::_base_/datasets/cifar10_bs16.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py'
]

model = dict(
    _scope_='mmrazor',
    type='DAFLDataFreeDistillation',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        bgr_to_rgb=True),
    architecture=dict(
        cfg_path='mmcls::resnet/resnet18_8xb16_cifar10.py', pretrained=False),
    teachers=dict(
        res34=dict(
            build_cfg=dict(
                cfg_path='mmcls::resnet/resnet34_8xb16_cifar10.py',
                pretrained=True),
            ckpt_path='/mnt/lustre/zhangzhongyu.vendor/gml/data/GML_data/gml_checkpoint/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth'
        ),
    ),
    generator=dict(
        type='DAFLGenerator',
        img_size=32,
        latent_dim=1000,
        hidden_channels=128),
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            res34_fc=dict(type='ModuleOutputs', source='res34.head.fc')),
        distill_losses=dict(
            loss_kl=dict(type='KLDivergence', tau=1, loss_weight=1)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='res34_fc')))),
    generator_distiller=dict(
        type='ConfigurableDistiller',
        teacher_recorders=dict(
            res34_neck_gap=dict(type='ModuleOutputs', source='res34.neck.gap'),
            res34_fc=dict(type='ModuleOutputs', source='res34.head.fc')),
        distill_losses=dict(
            loss_res34_oh=dict(type='OnehotLikeLoss', loss_weight=0.05),
            loss_res34_ie=dict(type='InformationEntropyLoss', loss_weight=5),
            loss_res34_ac=dict(type='ActivationLoss', loss_weight=0.01)),
        loss_forward_mappings=dict(
            loss_res34_oh=dict(
                preds_T=dict(from_student=False, recorder='res34_fc')),
            loss_res34_ie=dict(
                preds_T=dict(from_student=False, recorder='res34_fc')),
            loss_res34_ac=dict(
                preds_T=dict(from_student=False, recorder='res34_neck_gap')))))

find_unused_parameters = True

# optimizer
batch_size_per_gpu = 256

optim_wrapper = dict(
    _delete_=True,
    constructor='mmrazor.SeparateOptimWrapperConstructor',
    architecture=dict(optimizer=dict(type='AdamW', lr=1e-1)),
    generator=dict(optimizer=dict(type='AdamW', lr=1e-3)))

param_scheduler = dict(
    _delete_=True,
    architecture=dict(
        type='mmcv.StepLR',
        step=[100 * 120, 200 * 120],
        by_epoch=False,
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=0.0001,
    ),
    generator=dict(
        type='mmcv.FixedLR',
        by_epoch=False,
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=0.0001))

# train_cfg = dict(
#     _delete_=True,
#     type='mmengine.IterBasedTrainLoop',
#     max_iters=250 * 120)

train_cfg = dict(
    _delete_=True,
    type='mmrazor.DistributedIterBasedLoop',
    max_iters=250 * 120)
    # is_dynamic_ddp=True)

# dataset_type = 'CIFAR10'
# preprocess_cfg = dict(
#     # RGB format normalization parameters
#     mean=[125.307, 122.961, 113.8575],
#     std=[51.5865, 50.847, 51.255],
#     # loaded images are already RGB format
#     to_rgb=False)
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         'data/cifar10': "s3://PAT/datasets/Imagenet"})
# )

# train_pipeline = [
    # dict(type='LoadImageFromFile', file_client_args=file_client_args),
    # dict(type='RandomCrop', crop_size=32, padding=4),
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # dict(type='PackClsInputs'),
# ]

# test_pipeline = [
    # dict(type='LoadImageFromFile', file_client_args=file_client_args),
    # dict(type='PackClsInputs'),
# ]

# train_dataloader = dict(
#     batch_size=16,
#     num_workers=2,
#     dataset=dict(
#         type=dataset_type,
#         data_prefix='data/cifar10',
#         test_mode=False,
#         pipeline=train_pipeline),
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     persistent_workers=True,
# )

# val_dataloader = dict(
#     batch_size=16,
#     num_workers=2,
#     dataset=dict(
#         type=dataset_type,
#         data_prefix='data/cifar10/',
#         test_mode=True,
#         pipeline=test_pipeline),
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     persistent_workers=True,
# )
