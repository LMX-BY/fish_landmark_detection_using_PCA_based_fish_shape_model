# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

# cfg_re50 = {
#     'name': 'Resnet50',
#     'min_sizes': [[16, 32], [64, 128], [256, 512]],
#     'steps': [8, 16, 32],
#     'variance': [0.1, 0.2],
#     'clip': False,
#     'loc_weight': 2.0,
#     'gpu_train': True,
#     'batch_size': 20,
#     'ngpu': 1,
#     'epoch': 500,
#     'decay1': 70,
#     'decay2': 90,
#     'image_size': 840,
#     'pretrain': True,
#     'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
#     'in_channels_list': [512,1024,2048],
#     'in_channel': 256,
#     'out_channel': 256,
#     'prior_num_in_a_cell': 2
# }

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'prior_box_sizes': [[[55, 59], [55, 15], [55, 25]], [[94, 101], [94, 26], [94, 43]], [[140, 140], [140, 39], [140, 64]]],
    'steps': [8, 16, 32],
    'clip': False,
    'decay1': 70,
    'decay2': 90,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channels_list': [512,1024,2048],
    'in_channel': 256,
    'out_channel': 256,
    'prior_num_in_a_cell': 3
}

cfg_vgg16_2_layer_8_16 = {
    'name': 'vgg16',
    'pretrain': True,
    'prior_box_sizes': [[[93, 84], [62, 97], [44, 73], [64, 27], [142, 46], [77, 55], [128, 66], [104, 37]],
                        [[93, 84], [62, 97], [44, 73], [64, 27], [142, 46], [77, 55], [128, 66], [104, 37]]],
    'steps': [8, 16],
    'clip': False,
    'return_layers': {'16': 1, '23': 2},
    'in_channels_list': [256, 512],
    'in_channel': 256,
    'out_channel': 256,
    'landmark_dim': 8
}

cfg_vgg16_pca_2_layer = {
    'name': 'vgg16',
    'pretrain': True,
    'return_layers': {'23': 1, '30': 2},
    'in_channels_list': [512,512],
    'in_channel': 256,
    'out_channel': 256,
    'prior_num_in_a_cell': 16,
    'pca_feature_size': 6
}

cfg_vgg16_pca_3_layer = {
    'name': 'vgg16',
    'pretrain': True,
    'return_layers': {'16':3, '23': 2, '30': 32},
    'in_channels_list': [256, 512, 512],
    'in_channel': 256,
    'out_channel': 256,
    'prior_num_in_a_cell': 16,
    'pca_feature_size': 6
}

