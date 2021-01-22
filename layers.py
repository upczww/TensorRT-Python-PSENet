import numpy as np
import tensorrt as trt


def add_batchnorm(reader, network, input, lname, eps):
    gamma = reader.get_tensor(lname + "gamma")
    beta = reader.get_tensor(lname + "beta")
    mean = reader.get_tensor(lname + "moving_mean")
    var = reader.get_tensor(lname + "moving_variance")

    scale = gamma / np.sqrt(var + eps)
    shift = -mean / np.sqrt(var + eps) * gamma + beta
    power = np.ones(len(gamma), dtype=np.float32)

    bn = network.add_scale(
        input,
        trt.ScaleMode.CHANNEL,
        trt.Weights(shift),
        trt.Weights(scale),
        trt.Weights(power),
    )
    return bn


def bottleneck(reader, network, input, ch, stride, lname, branch_type):

    w = reader.get_tensor(lname + "conv1/weights").transpose(3, 2, 0, 1).reshape(-1)
    b = np.zeros(ch, dtype=np.float32)
    conv1 = network.add_convolution(input, ch, (1, 1), trt.Weights(w), trt.Weights(b))

    bn1 = add_batchnorm(reader, network, conv1.get_output(0), lname + "conv1/BatchNorm/", 1e-5)

    relu1 = network.add_activation(bn1.get_output(0), trt.ActivationType.RELU)

    w = reader.get_tensor(lname + "conv2/weights").transpose(3, 2, 0, 1).reshape(-1)
    b = np.zeros(ch, dtype=np.float32)
    conv2 = network.add_convolution(relu1.get_output(0), ch, (3, 3), trt.Weights(w), trt.Weights(b))
    conv2.stride = (stride, stride)
    conv2.padding = (1, 1)

    bn2 = add_batchnorm(reader, network, conv2.get_output(0), lname + "conv2/BatchNorm/", 1e-5)

    relu2 = network.add_activation(bn2.get_output(0), trt.ActivationType.RELU)

    w = reader.get_tensor(lname + "conv3/weights").transpose(3, 2, 0, 1).reshape(-1)
    b = np.zeros(ch * 4, dtype=np.float32)
    conv3 = network.add_convolution(relu2.get_output(0), ch * 4, (1, 1), trt.Weights(w), trt.Weights(b))

    bn3 = add_batchnorm(reader, network, conv3.get_output(0), lname + "conv3/BatchNorm/", 1e-5)

    # branch_type 0:shortcut,1:conv+bn+shortcut,2:maxpool+shortcut
    if branch_type == 0:
        ew1 = network.add_elementwise(input, bn3.get_output(0), trt.ElementWiseOperation.SUM)
    elif branch_type == 1:
        w = reader.get_tensor(lname + "shortcut/weights").transpose(3, 2, 0, 1).reshape(-1)
        b = np.zeros(ch * 4, dtype=np.float32)
        conv4 = network.add_convolution(input, ch * 4, (1, 1), trt.Weights(w), trt.Weights(b))
        conv4.stride = (stride, stride)
        bn4 = add_batchnorm(reader, network, conv4.get_output(0), lname + "shortcut/BatchNorm/", 1e-5)
        ew1 = network.add_elementwise(bn4.get_output(0), bn3.get_output(0), trt.ElementWiseOperation.SUM)
    else:
        pool = network.add_pooling(input, trt.PoolingType.MAX, (1, 1))
        pool.stride = (2, 2)
        ew1 = network.add_elementwise(pool.get_output(0), bn3.get_output(0), trt.ElementWiseOperation.SUM)

    relu3 = network.add_activation(ew1.get_output(0), trt.ActivationType.RELU)

    return relu3


def add_conv_relu(reader, network, input, outch, kernel, stride, lname):
    w = reader.get_tensor(lname + "weights").transpose(3, 2, 0, 1).reshape(-1)
    b = reader.get_tensor(lname + "biases")
    conv = network.add_convolution(input, outch, (kernel, kernel), trt.Weights(w), trt.Weights(b))
    conv.stride = (stride, stride)
    if kernel == 3:
        conv.padding = (1, 1)

    ac = network.add_activation(conv.get_output(0), trt.ActivationType.RELU)
    return ac

