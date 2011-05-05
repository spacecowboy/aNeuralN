import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
import time
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(1000).astype(numpy.float32)
b = numpy.random.randn(1000).astype(numpy.float32)

dest = numpy.zeros_like(a)
start = time.time()
multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400,1,1), grid=(1,1))
stop = time.time()
print("Cuda time: " + str(stop-start))

start = time.time()
dest_cpu = a*b
stop = time.time()
print("Cpu time: " + str(stop-start))

print dest-dest_cpu