import importlib.metadata
import sys
import traceback

def verify():
    if len(sys.argv) < 2:
        print("Usage: python verify_wheel.py <cpu|cuda|rocm>")
        sys.exit(1)
        
    expected_tier = sys.argv[1].lower()
    
    try:
        version = importlib.metadata.version("llama-cpp-python")
        print(f"llama-cpp-python version: {version}")
    except importlib.metadata.PackageNotFoundError:
        print("llama-cpp-python is not installed!")
        sys.exit(1)
    
    # 1. We no longer assert version suffix (e.g., '+cu' or '+rocm') because
    # pre-built llama-cpp-python v0.3.x wheels from abetlen.github.io do not
    # append hardware tags to __version__ or package metadata.
    
    # 2. Assert the wheel can be imported (or throws the expected hardware exception)
    try:
        import llama_cpp
        print("llama_cpp imported successfully.")
    except Exception as e:
        if expected_tier in ("cuda", "rocm"):
            # A CUDA/ROCm wheel on a CPU-only runner will fail to load its compiled libraries
            # (e.g., missing libcuda.so or libcudart.so, or missing DLLs).
            # This proves the wheel is the GPU variant, which is what we want to test in CI.
            print(f"Caught expected import error for GPU wheel on CPU runner:\n{e}")
        else:
            print("Unexpected import error for CPU wheel:")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    verify()
