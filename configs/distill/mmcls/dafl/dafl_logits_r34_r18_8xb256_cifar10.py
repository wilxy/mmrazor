_base_ = [
    'mmcls::_base_/datasets/cifar10_bs16.py',
    # 'mmcls::_base_/schedules/imagenet_bs256.py',
    'mmcls::_base_/default_runtime.py'
]

model = dict(
    _scope_='mmrazor',
    type='DAFLDataFreeStudentDistillation',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        bgr_to_rgb=True),
    architecture=dict(
        cfg_path='mmcls::resnet/resnet18_8xb16_cifar10.py', pretrained=False),
    teachers=[
        dict(
            name='resnet34',
            cfg=dict(cfg_path='mmcls::resnet/resnet34_8xb16_cifar10', pretrained=True),
            ckpt_path='/mnt/cache/zhangzhongyu/work_dir/checkpoints/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth'),
    ]
    generator=dict(
        type='DAFLGenerator',
        img_size=32,
        latent_dim=1000,
        hidden_channels=128),
    distiller_teacher_name='resnet34',
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_kl=dict(type='KLDivergence', tau=1, loss_weight=1)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')))),
    generator_distiller=dict(
        type='DataFreeDistiller',
        multi_teacher_cfgs=[
            dict(
                teacher_name='resnet34',
                teacher_recorders=dict(
                    neck_gap=dict(type='ModuleOutputs', source='neck.gap'),
                    fc=dict(type='ModuleOutputs', source='head.fc')),
                distill_losses=dict(
                    loss_res34_oh=dict(type='OnehotLikeLoss', loss_weight=0.05),
                    loss_res34_ie=dict(type='InformationEntropyLoss', loss_weight=5),
                    loss_res34_ac=dict(type='ActivationLoss', loss_weight=0.01)),
                loss_forward_mappings=dict(
                    loss_res34_oh=dict(
                        preds_T=dict(from_student=False, recorder='fc')),
                    loss_res34_ie=dict(
                        preds_T=dict(from_student=False, recorder='fc')),
                    loss_res34_ac=dict(
                        preds_T=dict(from_student=False, recorder='neck_gap'))))
        ]))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

# optimizer
batch_size_per_gpu = 256
# optimizer = dict(
#     model=dict(type='AdamW', lr=1e-1),
#     generator=dict(type='AdamW', lr=1e-3))
optim_wrapper = dict(
    _delete_=True,  # 不继承原始optim_wrapper这个dict的内容，即optim_wrapper就只有下面三行的内容
    constructor='mmrazor.SeparateOptimWrapperConstructor',
    architecture=dict(optimizer=dict(type='AdamW', lr=1e-1)),
    generator=dict(optimizer=dict(type='AdamW', lr=1e-3)))

# learning policy
lr_config = dict(
    policy='mmrazor.Multi',
    lr_updater_cfgs=dict(
        architecture=dict(
            policy='step',
            step=[100 * 120, 200 * 120],
            by_epoch=False,
            warmup='linear',
            warmup_iters=500,
            warmup_ratio=0.0001,
        ),
        generator=dict(
            policy='fixed',
            by_epoch=False,
            warmup='linear',
            warmup_iters=500,
            warmup_ratio=0.0001,
        )),
)

runner = dict(
    type='mmrazor.DynamicIterBasedRunner',
    max_iters=250 * 120,
    is_dynamic_ddp=True)
