import cuda.bindings.driver as cuda

def test_cuda_memcpy():
    # Init CUDA
    (err,) = cuda.cuInit(0)
    # assert err == cuda.CUresult.CUDA_SUCCESS

    # # Get device
    err, device = cuda.cuDeviceGet(0)
    # assert err == cuda.CUresult.CUDA_SUCCESS

    # # Construct context
    # err, ctx = cuda.cuCtxCreate(None, 0, device)
    # assert err == cuda.CUresult.CUDA_SUCCESS

    # # Allocate dev memory
    # size = int(1024 * np.uint8().itemsize)
    # err, dptr = cuda.cuMemAlloc(size)
    # assert err == cuda.CUresult.CUDA_SUCCESS

    # # Set h1 and h2 memory to be different
    # h1 = np.full(size, 1).astype(np.uint8)
    # h2 = np.full(size, 2).astype(np.uint8)
    # assert np.array_equal(h1, h2) is False

    # # h1 to D
    # (err,) = cuda.cuMemcpyHtoD(dptr, h1, size)
    # assert err == cuda.CUresult.CUDA_SUCCESS

    # # D to h2
    # (err,) = cuda.cuMemcpyDtoH(h2, dptr, size)
    # assert err == cuda.CUresult.CUDA_SUCCESS

    # # Validate h1 == h2
    # assert np.array_equal(h1, h2)

    # # Cleanup
    # (err,) = cuda.cuMemFree(dptr)
    # assert err == cuda.CUresult.CUDA_SUCCESS
    # (err,) = cuda.cuCtxDestroy(ctx)
    # assert err == cuda.CUresult.CUDA_SUCCESS

if __name__ == "__main__":
    test_cuda_memcpy()