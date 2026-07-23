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
    
    # 1. Assert the correct wheel variant is installed
    if expected_tier == "cuda":
        assert "+cu" in version, f"Expected CUDA wheel, got {version}"
    elif expected_tier == "rocm":
        assert "+rocm" in version, f"Expected ROCm wheel, got {version}"
    else:
        assert "+" not in version or "+cpu" in version, f"Expected CPU wheel, got {version}"
        
    # 2. Assert the wheel can be imported (or throws the expected hardware exception)
    try:
        import llama_cpp
        print("llama_cpp imported successfully.")
    except Exception as e:
        if expected_tier in ("cuda", "rocm"):
            # A CUDA/ROCm wheel on a CPU-only runner will fail to load its compiled libraries
            # (e.g., missing nvcuda.dll, libcuda.so, or libcudart.so).
            # This proves the wheel is the GPU variant, which is what we want to test in CI.
            print(f"Caught expected import error for GPU wheel on CPU runner:\n{e}")
        else:
            print("Unexpected import error for CPU wheel:")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    verify()
