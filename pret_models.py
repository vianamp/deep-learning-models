from keras import layers
from keras import optimizers
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import SeparableConv2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.utils.data_utils import get_file

#
# VGG16
#

def Get_VGG16(n_classes = 5):

	img_input = Input(shape=(None,None,3))

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

	base_model = Model(img_input, x, name='vgg16')

	weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', cache_subdir='models')
	base_model.load_weights(weights_path)

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	predictions = Dense(n_classes, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions, name='vgg16-adapted')

	for layer in base_model.layers:
		layer.trainable = False

	return model

#
# ResNet 50
#

def identity_block(input_tensor, kernel_size, filters, stage, block):

	filters1, filters2, filters3 = filters

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

	x = layers.add([x, input_tensor])
	x = Activation('relu')(x)
	return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

	filters1, filters2, filters3 = filters

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size, padding='same',
			   name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

	shortcut = Conv2D(filters3, (1, 1), strides=strides,
					  name=conv_name_base + '1')(input_tensor)
	shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

	x = layers.add([x, shortcut])
	x = Activation('relu')(x)
	return x

def Get_ResNet50(im_size=224, n_classes=5):

	img_input = Input(shape=(im_size,im_size,3))

	x = ZeroPadding2D((3, 3))(img_input)
	x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
	x = BatchNormalization(axis=3, name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

	x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

	x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

	x = AveragePooling2D((7, 7), name='avg_pool')(x)

	base_model = Model(img_input, x, name='resnet50')

	weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af')
	base_model.load_weights(weights_path)

	x = base_model.output
	x = Flatten()(x)
	predictions = Dense(n_classes, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions, name='resnet50-adapted')

	for layer in base_model.layers:
		layer.trainable = False

	return model

#
# Xception
#

def Get_Xception(im_size=224, n_classes=5):

	img_input = Input(shape=(im_size,im_size,3))

	x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
	x = BatchNormalization(name='block1_conv1_bn')(x)
	x = Activation('relu', name='block1_conv1_act')(x)
	x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
	x = BatchNormalization(name='block1_conv2_bn')(x)
	x = Activation('relu', name='block1_conv2_act')(x)

	residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
	residual = BatchNormalization()(residual)

	x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
	x = BatchNormalization(name='block2_sepconv1_bn')(x)
	x = Activation('relu', name='block2_sepconv2_act')(x)
	x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
	x = BatchNormalization(name='block2_sepconv2_bn')(x)

	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
	x = layers.add([x, residual])

	residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
	residual = BatchNormalization()(residual)

	x = Activation('relu', name='block3_sepconv1_act')(x)
	x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
	x = BatchNormalization(name='block3_sepconv1_bn')(x)
	x = Activation('relu', name='block3_sepconv2_act')(x)
	x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
	x = BatchNormalization(name='block3_sepconv2_bn')(x)

	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
	x = layers.add([x, residual])

	residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
	residual = BatchNormalization()(residual)

	x = Activation('relu', name='block4_sepconv1_act')(x)
	x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
	x = BatchNormalization(name='block4_sepconv1_bn')(x)
	x = Activation('relu', name='block4_sepconv2_act')(x)
	x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
	x = BatchNormalization(name='block4_sepconv2_bn')(x)

	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
	x = layers.add([x, residual])

	for i in range(8):
		residual = x
		prefix = 'block' + str(i + 5)

		x = Activation('relu', name=prefix + '_sepconv1_act')(x)
		x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
		x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
		x = Activation('relu', name=prefix + '_sepconv2_act')(x)
		x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
		x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
		x = Activation('relu', name=prefix + '_sepconv3_act')(x)
		x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
		x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

		x = layers.add([x, residual])

	residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
	residual = BatchNormalization()(residual)

	x = Activation('relu', name='block13_sepconv1_act')(x)
	x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
	x = BatchNormalization(name='block13_sepconv1_bn')(x)
	x = Activation('relu', name='block13_sepconv2_act')(x)
	x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
	x = BatchNormalization(name='block13_sepconv2_bn')(x)

	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
	x = layers.add([x, residual])

	x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
	x = BatchNormalization(name='block14_sepconv1_bn')(x)
	x = Activation('relu', name='block14_sepconv1_act')(x)

	x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
	x = BatchNormalization(name='block14_sepconv2_bn')(x)
	x = Activation('relu', name='block14_sepconv2_act')(x)

	base_model = Model(img_input, x, name='xception')

	weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5', 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5', cache_subdir='models')
	base_model.load_weights(weights_path)

	x = base_model.output

	#x = GlobalAveragePooling2D()(x)
	#x = Dense(1024, activation='relu')(x)

	x = Flatten()(x)

	predictions = Dense(n_classes, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions, name='xception-adapted')

	for layer in base_model.layers:
		layer.trainable = False

	return model

#
# Inception V3
#

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):

	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None
	x = Conv2D(
		filters, (num_row, num_col),
		strides=strides,
		padding=padding,
		use_bias=False,
		name=conv_name)(x)
	x = BatchNormalization(axis=3, scale=False, name=bn_name)(x)
	x = Activation('relu', name=name)(x)
	return x

def Get_InceptionV3(im_size=224, n_classes=5):

	img_input = Input(shape=(im_size,im_size,3))

	x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
	x = conv2d_bn(x, 32, 3, 3, padding='valid')
	x = conv2d_bn(x, 64, 3, 3)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	x = conv2d_bn(x, 80, 1, 1, padding='valid')
	x = conv2d_bn(x, 192, 3, 3, padding='valid')
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	# mixed 0, 1, 2: 35 x 35 x 256
	branch1x1 = conv2d_bn(x, 64, 1, 1)

	branch5x5 = conv2d_bn(x, 48, 1, 1)
	branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

	branch3x3dbl = conv2d_bn(x, 64, 1, 1)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
	x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed0')

	# mixed 1: 35 x 35 x 256
	branch1x1 = conv2d_bn(x, 64, 1, 1)

	branch5x5 = conv2d_bn(x, 48, 1, 1)
	branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

	branch3x3dbl = conv2d_bn(x, 64, 1, 1)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
	x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed1')

	# mixed 2: 35 x 35 x 256
	branch1x1 = conv2d_bn(x, 64, 1, 1)

	branch5x5 = conv2d_bn(x, 48, 1, 1)
	branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

	branch3x3dbl = conv2d_bn(x, 64, 1, 1)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
	x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed2')

	# mixed 3: 17 x 17 x 768
	branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

	branch3x3dbl = conv2d_bn(x, 64, 1, 1)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

	branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
	x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

	# mixed 4: 17 x 17 x 768
	branch1x1 = conv2d_bn(x, 192, 1, 1)

	branch7x7 = conv2d_bn(x, 128, 1, 1)
	branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
	branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

	branch7x7dbl = conv2d_bn(x, 128, 1, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
	x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],axis=3,name='mixed4')

	# mixed 5, 6: 17 x 17 x 768
	for i in range(2):
		branch1x1 = conv2d_bn(x, 192, 1, 1)

		branch7x7 = conv2d_bn(x, 160, 1, 1)
		branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
		branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

		branch7x7dbl = conv2d_bn(x, 160, 1, 1)
		branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
		branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
		branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
		branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

		branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
		branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
		x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3,name='mixed' + str(5 + i))

	# mixed 7: 17 x 17 x 768
	branch1x1 = conv2d_bn(x, 192, 1, 1)

	branch7x7 = conv2d_bn(x, 192, 1, 1)
	branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
	branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

	branch7x7dbl = conv2d_bn(x, 192, 1, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
	x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed7')

	# mixed 8: 8 x 8 x 1280
	branch3x3 = conv2d_bn(x, 192, 1, 1)
	branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

	branch7x7x3 = conv2d_bn(x, 192, 1, 1)
	branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
	branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
	branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

	branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
	x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

	# mixed 9: 8 x 8 x 2048
	for i in range(2):
		branch1x1 = conv2d_bn(x, 320, 1, 1)

		branch3x3 = conv2d_bn(x, 384, 1, 1)
		branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
		branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
		branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

		branch3x3dbl = conv2d_bn(x, 448, 1, 1)
		branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
		branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
		branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
		branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

		branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
		branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
		x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed' + str(9 + i))

	base_model = Model(img_input, x, name='inceptionv3')

	weights_path = get_file('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', cache_subdir='models')
	base_model.load_weights(weights_path)

	x = base_model.output
	x = Flatten()(x)

	predictions = Dense(n_classes, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions, name='inceptionv3-adapted')

	for layer in base_model.layers:
		layer.trainable = False

	return model

	# if include_top:
	#     # Classification block
	#     x = GlobalAveragePooling2D(name='avg_pool')(x)
	#     x = Dense(classes, activation='softmax', name='predictions')(x)
	# else:
	#     if pooling == 'avg':
	#         x = GlobalAveragePooling2D()(x)
	#     elif pooling == 'max':
	#         x = GlobalMaxPooling2D()(x)

	# # Ensure that the model takes into account
	# # any potential predecessors of `input_tensor`.
	# if input_tensor is not None:
	#     inputs = get_source_inputs(input_tensor)
	# else:
	#     inputs = img_input
	# # Create model.
	# model = Model(inputs, x, name='inception_v3')

	# # load weights
	# if weights == 'imagenet':
	#     if K.image_data_format() == 'channels_first':
	#         if K.backend() == 'tensorflow':
	#             warnings.warn('You are using the TensorFlow backend, yet you '
	#                           'are using the Theano '
	#                           'image data format convention '
	#                           '(`image_data_format="channels_first"`). '
	#                           'For best performance, set '
	#                           '`image_data_format="channels_last"` in '
	#                           'your Keras config '
	#                           'at ~/.keras/keras.json.')
	#     if include_top:
	#         weights_path = get_file(
	#             'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
	#             WEIGHTS_PATH,
	#             cache_subdir='models',
	#             md5_hash='9a0d58056eeedaa3f26cb7ebd46da564')
	#     else:
	#         weights_path = get_file(
	#             'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
	#             WEIGHTS_PATH_NO_TOP,
	#             cache_subdir='models',
	#             md5_hash='bcbd6486424b2319ff4ef7d526e38f63')
	#     model.load_weights(weights_path)
	#     if K.backend() == 'theano':
	#         convert_all_kernels_in_model(model)
	# return model


def Get_PreTrainedModel(model_name, im_size, n_classes):

	if model_name == 'vgg16':

		model = Get_VGG16(n_classes = n_classes)

	if model_name == 'resnet50':

		model = Get_ResNet50(im_size=im_size, n_classes = n_classes)

	if model_name == 'xception':

		model = Get_Xception(im_size=im_size, n_classes = n_classes)

	if model_name == 'inceptionv3':

		model = Get_InceptionV3(im_size=im_size, n_classes = n_classes)

	sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

	return model
