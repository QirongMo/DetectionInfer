
import pycuda.driver as cuda
import tensorrt as trt

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, shape=None):
        self.host = host_mem
        self.device = device_mem
        self.shape = shape

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine, context=None, input_shape=None):
    inputs, outputs, bindings = [], [], [] 
    for i in range(engine.num_bindings):
        name = engine.get_tensor_name(i)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = engine.get_tensor_shape(name)
        # print(name, dtype, shape)  
        is_input = (engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT)
        if shape[-1] == -1:
            if is_input:
                shape[-2], shape[-1] = input_shape
                context.set_input_shape(name, shape)
            shape = context.get_tensor_shape(name) # 再次获取shape，使用context获取能获取本次input_shape下的真正shape
        size = trt.volume(shape)#  * engine.max_batch_size#The maximum batch size which can be used for inference.
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        if is_input: # Determine whether a binding is an input binding.
            inputs.append(HostDeviceMem(host_mem, device_mem, shape))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, shape))       
         
    return inputs, outputs, bindings

