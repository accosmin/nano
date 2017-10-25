import models

outlayer = make_output_layer(omaps = 10)

normlayer = make_norm_by_plane_layer()

mlp0 = "--model forward-network --model-params "
mlp1 = mlp0 + make_affine_layer(omaps = 128)
mlp2 = mlp1 + make_affine_layer(omaps = 128)
mlp3 = mlp2 + make_affine_layer(omaps = 128)
mlp4 = mlp3 + make_affine_layer(omaps = 128)
mlp5 = mlp4 + make_affine_layer(omaps = 128)
mlp6 = mlp5 + make_affine_layer(omaps = 128)
mlp7 = mlp6 + make_affine_layer(omaps = 128)
mlp8 = mlp7 + make_affine_layer(omaps = 128)

convnet0 = "--model forward-network --model-params "
convnet1 = convnet0 + make_conv3d_layer(omaps = 32, krows = 9, kcols = 9)
convnet2 = convnet1 + make_conv3d_layer(omaps = 48, krows = 7, kcols = 7)
convnet3 = convnet2 + make_conv3d_layer(omaps = 64, krows = 5, kcols = 5)
convnet4 = convnet3 + make_conv3d_layer(omaps = 64, krows = 5, kcols = 5)
convnet5 = convnet4 + make_conv3d_layer(omaps = 96, krows = 3, kcols = 3)
convnet6 = convnet5 + make_conv3d_layer(omaps = 96, krows = 3, kcols = 3)
convnet7 = convnet6 + make_conv3d_layer(omaps = 96, krows = 3, kcols = 3)
convnet8 = convnet7 + make_conv3d_layer(omaps = 128, krows = 1, kcols = 1)
