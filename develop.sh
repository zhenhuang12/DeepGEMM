# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Link CUTLASS includes
ln -sf $script_dir/third-party/cutlass/include/cutlass deep_gemm/include
ln -sf $script_dir/third-party/cutlass/include/cute deep_gemm/include

# Remove old dist file, build files, and build
rm -rf build dist
rm -rf *.egg-info
python setup.py build

# Find the .so file in build directory and create symlink in current directory
if python -c "import torch; raise SystemExit(0 if torch.version.hip else 1)"; then
    echo "ROCm build detected, skipping extension symlink"
else
    so_file=$(python -c "import glob; files = glob.glob('build/**/*.so', recursive=True); print(files[0] if files else '')")
    if [ -n "$so_file" ]; then
        ln -sf "../$so_file" deep_gemm/
    else
        echo "Error: No SO file found in build directory" >&2
        exit 1
    fi
fi

# Open users' original directory
cd "$original_dir"
