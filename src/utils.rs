use std::fs;
use std::path::Path;

use candle_core::{DType, Device};

pub fn get_device() -> anyhow::Result<Device> {
    #[cfg(feature = "cuda")]
    {
        // Trying to get CUDA device
        if let Ok(device) = Device::new_cuda(0) {
            return Ok(device);
        }
        println!("Warning: CUDA is unavailable, fallback to CPU.");
    }

    #[cfg(feature = "metal")]
    {
        // Trying to get Metal device
        if let Ok(device) = Device::new_metal(0) {
            return Ok(device);
        }
        println!("Warning: Metal is unavailable, fallback to CPU.");
    }

    // Default CPU fallback
    Ok(Device::Cpu)
}

pub fn get_device_dtype(device: &Device) -> anyhow::Result<DType> {
    match device {
        Device::Cpu => Ok(DType::F32), // F32 is better for CPU
        Device::Metal(_) => Ok(DType::BF16),
        Device::Cuda(_) => Ok(DType::BF16), // Not sure how to better manage F16 (up to RTX3000) and BF16
    }
}

pub fn resolve_input(s: &str) -> String {
    let path = Path::new(s);
    if path.exists() && path.is_file() {
        fs::read_to_string(path).unwrap_or_else(|_| s.to_string())
    } else {
        s.to_string()
    }
}

pub fn resolve_path(s: &str) -> String {
    let path = Path::new(s);
    if path.exists() {
        s.to_string()
    } else {
        panic!("File not found: {}", s);
    }
}

pub fn is_directory(s: &str) -> bool {
    Path::new(s).is_dir()
}

pub fn get_files_from_directory(dir: &str) -> Vec<String> {
    let path = Path::new(dir);
    if !path.is_dir() {
        return vec![dir.to_string()];
    }

    let mut files: Vec<String> = fs::read_dir(path)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_file())
        .map(|entry| entry.path().to_string_lossy().to_string())
        .collect();

    files.sort();
    files
}
