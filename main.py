import argparse
import os
import time

import cv2
import numpy as np

# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit
import pycuda.driver as cuda
import tensorflow as tf
import tensorrt as trt
from tensorflow.python import pywrap_tensorflow

import common
from layers import add_batchnorm, add_conv_relu, bottleneck
from utils import draw_result, postprocess, preprocess

INPUT_NAME = "input"
OUTPUT_NAME = "output"
USE_FP16 = False
VERBOSE = False
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if VERBOSE else trt.Logger(trt.Logger.WARNING)

# init tensorrt plugins if necessary
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def build_engine(model_dir):
    """Build TensorRT engine through the Python API.
    Args:
        model_dir: the trained TensorFlow PSENet model dir.

    Returns:
        engine: the build TensorRT engine.
    """
    ckpt = tf.train.get_checkpoint_state(model_dir)
    ckpt_path = ckpt.model_checkpoint_path
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    explicit_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        explicit_flag
    ) as network, builder.create_builder_config() as config:
        data = network.add_input(INPUT_NAME, trt.float32, (-1, 3, -1, -1))

        w = reader.get_tensor("resnet_v1_50/conv1/weights").transpose(3, 2, 0, 1).reshape(-1)
        b = np.zeros(64, dtype=np.float32)
        conv1 = network.add_convolution(data, 64, (7, 7), trt.Weights(w), trt.Weights(b))
        conv1.stride = (2, 2)
        conv1.padding = (3, 3)

        bn1 = add_batchnorm(reader, network, conv1.get_output(0), "resnet_v1_50/conv1/BatchNorm/", 1e-5)
        relu1 = network.add_activation(bn1.get_output(0), trt.ActivationType.RELU)

        # C2
        pool1 = network.add_pooling(relu1.get_output(0), trt.PoolingType.MAX, (3, 3))
        pool1.stride = (2, 2)
        pool1.pre_padding = (0, 0)
        pool1.post_padding = (1, 1)

        x = bottleneck(reader, network, pool1.get_output(0), 64, 1, "resnet_v1_50/block1/unit_1/bottleneck_v1/", 1)

        x = bottleneck(reader, network, x.get_output(0), 64, 1, "resnet_v1_50/block1/unit_2/bottleneck_v1/", 0)

        # C3
        block1 = bottleneck(reader, network, x.get_output(0), 64, 2, "resnet_v1_50/block1/unit_3/bottleneck_v1/", 2)

        x = bottleneck(reader, network, block1.get_output(0), 128, 1, "resnet_v1_50/block2/unit_1/bottleneck_v1/", 1)
        x = bottleneck(reader, network, x.get_output(0), 128, 1, "resnet_v1_50/block2/unit_2/bottleneck_v1/", 0)
        x = bottleneck(reader, network, x.get_output(0), 128, 1, "resnet_v1_50/block2/unit_3/bottleneck_v1/", 0)
        # C4
        block2 = bottleneck(reader, network, x.get_output(0), 128, 2, "resnet_v1_50/block2/unit_4/bottleneck_v1/", 2)

        x = bottleneck(reader, network, block2.get_output(0), 256, 1, "resnet_v1_50/block3/unit_1/bottleneck_v1/", 1)
        x = bottleneck(reader, network, x.get_output(0), 256, 1, "resnet_v1_50/block3/unit_2/bottleneck_v1/", 0)
        x = bottleneck(reader, network, x.get_output(0), 256, 1, "resnet_v1_50/block3/unit_3/bottleneck_v1/", 0)
        x = bottleneck(reader, network, x.get_output(0), 256, 1, "resnet_v1_50/block3/unit_4/bottleneck_v1/", 0)
        x = bottleneck(reader, network, x.get_output(0), 256, 1, "resnet_v1_50/block3/unit_5/bottleneck_v1/", 0)
        block3 = bottleneck(reader, network, x.get_output(0), 256, 2, "resnet_v1_50/block3/unit_6/bottleneck_v1/", 2)

        x = bottleneck(reader, network, block3.get_output(0), 512, 1, "resnet_v1_50/block4/unit_1/bottleneck_v1/", 1)
        x = bottleneck(reader, network, x.get_output(0), 512, 1, "resnet_v1_50/block4/unit_2/bottleneck_v1/", 0)
        # C5
        block4 = bottleneck(reader, network, x.get_output(0), 512, 1, "resnet_v1_50/block4/unit_3/bottleneck_v1/", 0)

        build_p5_r1 = add_conv_relu(reader, network, block4.get_output(0), 256, 1, 1, "build_feature_pyramid/build_P5/")

        build_p4_r1 = add_conv_relu(
            reader, network, block2.get_output(0), 256, 1, 1, "build_feature_pyramid/build_P4/reduce_dimension/"
        )

        bfp_layer4_resize = network.add_resize(build_p5_r1.get_output(0))
        build_p4_r1_shape = network.add_shape(build_p4_r1.get_output(0)).get_output(0)
        bfp_layer4_resize.set_input(1, build_p4_r1_shape)
        bfp_layer4_resize.resize_mode = trt.ResizeMode.NEAREST
        bfp_layer4_resize.align_corners = False

        bfp_add = network.add_elementwise(
            build_p4_r1.get_output(0), bfp_layer4_resize.get_output(0), trt.ElementWiseOperation.SUM
        )

        build_p4_r2 = add_conv_relu(
            reader, network, bfp_add.get_output(0), 256, 3, 1, "build_feature_pyramid/build_P4/avoid_aliasing/"
        )

        build_p3_r1 = add_conv_relu(
            reader, network, block1.get_output(0), 256, 1, 1, "build_feature_pyramid/build_P3/reduce_dimension/"
        )

        bfp_layer3_resize = network.add_resize(build_p4_r2.get_output(0))
        bfp_layer3_resize.resize_mode = trt.ResizeMode.NEAREST
        build_p3_r1_shape = network.add_shape(build_p3_r1.get_output(0)).get_output(0)
        bfp_layer3_resize.set_input(1, build_p3_r1_shape)
        bfp_layer3_resize.align_corners = False

        bfp_add1 = network.add_elementwise(
            bfp_layer3_resize.get_output(0), build_p3_r1.get_output(0), trt.ElementWiseOperation.SUM
        )

        build_p3_r2 = add_conv_relu(
            reader, network, bfp_add1.get_output(0), 256, 3, 1, "build_feature_pyramid/build_P3/avoid_aliasing/"
        )

        build_p2_r1 = add_conv_relu(
            reader, network, pool1.get_output(0), 256, 1, 1, "build_feature_pyramid/build_P2/reduce_dimension/"
        )

        bfp_layer2_resize = network.add_resize(build_p3_r2.get_output(0))
        bfp_layer2_resize.resize_mode = trt.ResizeMode.NEAREST
        build_p2_r1_shape = network.add_shape(build_p2_r1.get_output(0)).get_output(0)
        bfp_layer2_resize.set_input(1, build_p2_r1_shape)
        bfp_layer2_resize.align_corners = False

        bfp_add2 = network.add_elementwise(
            bfp_layer2_resize.get_output(0), build_p2_r1.get_output(0), trt.ElementWiseOperation.SUM
        )

        # P2
        build_p2_r2 = add_conv_relu(
            reader, network, bfp_add2.get_output(0), 256, 3, 1, "build_feature_pyramid/build_P2/avoid_aliasing/"
        )
        build_p2_r2_shape = network.add_shape(build_p2_r2.get_output(0)).get_output(0)

        # P3 x2
        layer1_resize = network.add_resize(build_p3_r2.get_output(0))
        layer1_resize.resize_mode = trt.ResizeMode.LINEAR
        layer1_resize.set_input(1, build_p2_r2_shape)
        layer1_resize.align_corners = False

        # P4 x4
        layer2_resize = network.add_resize(build_p4_r2.get_output(0))
        layer2_resize.resize_mode = trt.ResizeMode.LINEAR
        layer2_resize.set_input(1, build_p2_r2_shape)
        layer2_resize.align_corners = False

        # p5 right
        # P5 x8
        layer3_resize = network.add_resize(build_p5_r1.get_output(0))
        layer3_resize.resize_mode = trt.ResizeMode.LINEAR
        layer3_resize.set_input(1, build_p2_r2_shape)
        layer3_resize.align_corners = False

        # C(P5,P4,P3,P2)
        concat = network.add_concatenation(
            [
                layer3_resize.get_output(0),
                layer2_resize.get_output(0),
                layer1_resize.get_output(0),
                build_p2_r2.get_output(0),
            ]
        )

        w = reader.get_tensor("feature_results/Conv/weights").transpose(3, 2, 0, 1).reshape(-1)
        b = np.zeros(256, dtype=np.float32)
        feature_result_conv = network.add_convolution(concat.get_output(0), 256, (3, 3), trt.Weights(w), trt.Weights(b))
        feature_result_conv.padding = (1, 1)

        feature_result_bn = add_batchnorm(
            reader, network, feature_result_conv.get_output(0), "feature_results/Conv/BatchNorm/", 1e-5
        )

        feature_result_relu = network.add_activation(feature_result_bn.get_output(0), trt.ActivationType.RELU)
        w = reader.get_tensor("feature_results/Conv_1/weights").transpose(3, 2, 0, 1).reshape(-1)
        b = reader.get_tensor("feature_results/Conv_1/biases")
        feature_result_conv_1 = network.add_convolution(
            feature_result_relu.get_output(0), 6, (1, 1), trt.Weights(w), trt.Weights(b)
        )

        sigmoid = network.add_activation(feature_result_conv_1.get_output(0), trt.ActivationType.SIGMOID)
        sigmoid.get_output(0).name = OUTPUT_NAME
        network.mark_output(sigmoid.get_output(0))

        profile = builder.create_optimization_profile()
        profile.set_shape("input", min=(1, 3, 128, 128), opt=(1, 3, 640, 640), max=(4, 3, 1200, 1200))
        config.add_optimization_profile(profile)

        config.max_workspace_size = common.GiB(1)
        if USE_FP16:
            config_flags = 1 << int(trt.BuilderFlag.FP16)
            config.flags = config_flags

        engine = builder.build_engine(network, config)

        return engine


def serialize_engine(engine, engine_path):
    """Serialize engine.
    Args:
        engine: trt.ICudaEngine to be serialized.
        engine_path: where to save serialized engine.

    Returns:
        None
    """
    TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
    serialized_engine = engine.serialize()
    TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(engine_path))
    with open(engine_path, "wb") as fout:
        fout.write(serialized_engine)
    TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")


def get_engine(engine_path, model_dir):
    """Serialize engine.
    Args:
        engine_path: where to save serialized engine.
        model_dir: the trained TensorFlow PSENet model dir.

    Returns:
        engine: trt.ICudaEngine.
    """
    if os.path.exists(engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    else:
        engine = build_engine(model_dir)
        serialize_engine(engine, engine_path)
    return engine


def main(args):
    with get_engine(args.engine_path, args.model_dir) as engine:
        with engine.create_execution_context() as context:
            origin_img = cv2.imread(args.image_path)
            t1 = time.time()
            img, (ratio_h, ratio_w) = preprocess(origin_img)
            cv2.imwrite("processed.jpg", img)
            h, w, _ = img.shape
            # hwc to chw
            img = img.transpose((2, 0, 1))
            # flatten the image into a 1D array
            img = img.ravel()
            context.set_binding_shape(0, (1, 3, h, w))
            # allocate buffers and create a stream.
            inputs, outputs, bindings, stream = common.allocate_buffers(engine, context)
            # copy to pagelocked memory
            np.copyto(inputs[0].host, img)
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            # reshape 1D array to chw
            output = np.reshape(output, (6, h // 4, w // 4))
            # transpose chw to hwc
            output = output.transpose(1, 2, 0)
            boxes = postprocess(origin_img, output, ratio_h, ratio_w)
            t2 = time.time()
            print("total cost %fms" % ((t2 - t1) * 1000))
            draw_result(origin_img, boxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build engine and do inference")
    parser.add_argument("--model_dir", "-m", type=str, required=True, help="tensorflow model dir")
    parser.add_argument("--engine_path", "-e", type=str, required=True, help="engine file path")
    parser.add_argument("--image_path", "-i", type=str, required=True, help="test image path")

    args = parser.parse_args()
    # if os.path.exists(args.engine_path):
    #     os.remove(args.engine_path)
    main(args)
