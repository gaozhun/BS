import albumentations as albu
transform_dict = {
    "None": [  # 不增强
        albu.Rotate(p=0),
    ],
    "HorizontalFlip": [ # 只左右镜像
        albu.HorizontalFlip(p=1),
    ],
    "VerticalFlip": [ # 只上下镜像
        albu.VerticalFlip(p=1),
    ],
    "RandomRotate90": [ # 旋转0、90、180、270
        albu.RandomRotate90(p=1),
    ],
    "RandomRotate90_Flip": [ # 旋转+上下左右
        albu.HorizontalFlip(p=0.5),
        albu.RandomRotate90(p=1),
    ],
    "RandomRotate": [ # 随机旋转
        albu.Affine(rotate=(0,360), p=1),
    ]
}

# 对比实验
same_transform = "None"
obj_transform_dict = {
        'capsule': same_transform,
        'bottle': same_transform,
        'carpet': same_transform,
        'leather': same_transform,
        'pill': same_transform,
        'transistor': same_transform,
        'tile': same_transform,
        'cable': same_transform,
        'zipper': same_transform,
        'toothbrush': same_transform,
        'metal_nut': same_transform,
        'hazelnut': same_transform,
        'screw': same_transform,
        'grid': same_transform,
        'wood': same_transform,
}
# 推荐的
# obj_transform_dict = {
#         'capsule': same_transform,
#         'bottle': same_transform,
#         'carpet': same_transform,
#         'leather': same_transform,
#         'pill': same_transform,
#         'transistor': same_transform,
#         'tile': same_transform,
#         'cable': same_transform,
#         'zipper': same_transform,
#         'toothbrush': same_transform,
#         'metal_nut': same_transform,
#         'hazelnut': same_transform,
#         'screw': same_transform,
#         'grid': same_transform,
#         'wood': same_transform, 
# }