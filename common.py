import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def GiB(val):
    """Calculcate size in G bytes.
    Args:
        val: how many GiBs.
    Returns:
        calculated size.
    """
    return val * 1 << 30


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple.
    Attributes:
        host: host memory.
        device: device memory.
    """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, context):
    """Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    Args:
        engine: trt.ICudaEngine.
        context: TensorRT execution context.
    Returns:
        inputs: input buffers.
        outputs: outputs buffers.
        bindings: memory bindings.
        stream: TensorRT CUDA stream.
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(context.get_binding_shape(engine[binding]))  # * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """This function is generalized for multiple inputs/outputs.
       inputs and outputs are expected to be lists of HostDeviceMem objects.
    Args:
        context: TensorRT execution context.
        bindings: memory bindings.
        inputs: input buffers.
        outputs: outputs buffers.
        stream: TensorRT CUDA stream.
        batch_size: batch size.
    Returns:
        network outputs.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def do_inference_v2(context, bindings, inputs, outputs, stream):
    """This function is generalized for multiple inputs/outputs for full dimension networks.
       inputs and outputs are expected to be lists of HostDeviceMem objects.
    Args:
        context: TensorRT execution context.
        bindings: memory bindings.
        inputs: input buffers.
        outputs: outputs buffers.
        stream: TensorRT CUDA stream.
    Returns:
        network outputs.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
