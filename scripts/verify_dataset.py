#!/usr/bin/env python3
"""
Dataset verification script for animal classification project.

This script:
- Checks file counts per class
- Verifies image integrity (can be opened and read)
- Validates expected image dimensions (224x224x3)
- Identifies corrupted files
- Generates resolution distribution statistics
- Saves summary stats to reports/dataset_stats.json
"""

import os
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: Required libraries not found. Please install:")
    print("pip install Pillow numpy")
    sys.exit(1)


def verify_image(image_path: Path) -> Tuple[bool, Tuple[int, int, int], str]:
    """
    Verify if an image can be opened and get its dimensions.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (is_valid, dimensions, error_message)
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary to ensure 3 channels
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            width, height = img.size
            channels = len(img.getbands())
            
            return True, (height, width, channels), ""
            
    except Exception as e:
        return False, (0, 0, 0), str(e)


def get_class_directories(data_path: Path) -> List[Path]:
    """Get all class directories from the data/raw path."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    class_dirs.sort()  # Sort for consistent ordering
    
    return class_dirs


def verify_dataset(data_root: str = "data/raw") -> Dict[str, Any]:
    """
    Verify the entire dataset and generate statistics.
    
    Args:
        data_root: Root directory containing class folders
        
    Returns:
        Dictionary containing verification statistics
    """
    data_path = Path(data_root)
    
    # Initialize statistics
    stats = {
        "dataset_path": str(data_path.absolute()),
        "total_classes": 0,
        "total_images": 0,
        "total_corrupted": 0,
        "expected_size": [224, 224, 3],
        "classes": {},
        "corrupted_files": [],
        "resolution_distribution": {},
        "size_compliance": {
            "correct_size": 0,
            "incorrect_size": 0,
            "percentage_correct": 0.0
        }
    }
    
    print(f"Verifying dataset at: {data_path.absolute()}")
    print("=" * 60)
    
    try:
        class_dirs = get_class_directories(data_path)
        stats["total_classes"] = len(class_dirs)
        
        if not class_dirs:
            print("Warning: No class directories found!")
            return stats
        
        resolution_counter = Counter()
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            print(f"Processing class: {class_name}")
            
            # Get all jpg/jpeg files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
                image_files.extend(class_dir.glob(ext))
            
            class_stats = {
                "total_files": len(image_files),
                "valid_images": 0,
                "corrupted_images": 0,
                "correct_size_images": 0,
                "resolution_distribution": Counter()
            }
            
            for img_file in image_files:
                is_valid, dimensions, error_msg = verify_image(img_file)
                
                if is_valid:
                    class_stats["valid_images"] += 1
                    resolution_key = f"{dimensions[0]}x{dimensions[1]}x{dimensions[2]}"
                    class_stats["resolution_distribution"][resolution_key] += 1
                    resolution_counter[resolution_key] += 1
                    
                    # Check if image matches expected size
                    if list(dimensions) == stats["expected_size"]:
                        class_stats["correct_size_images"] += 1
                        stats["size_compliance"]["correct_size"] += 1
                    else:
                        stats["size_compliance"]["incorrect_size"] += 1
                        
                else:
                    class_stats["corrupted_images"] += 1
                    stats["corrupted_files"].append({
                        "file": str(img_file),
                        "class": class_name,
                        "error": error_msg
                    })
            
            # Convert Counter to regular dict for JSON serialization
            class_stats["resolution_distribution"] = dict(class_stats["resolution_distribution"])
            
            stats["classes"][class_name] = class_stats
            stats["total_images"] += class_stats["total_files"]
            stats["total_corrupted"] += class_stats["corrupted_images"]
            
            # Print class summary
            print(f"  - Files found: {class_stats['total_files']}")
            print(f"  - Valid images: {class_stats['valid_images']}")
            print(f"  - Corrupted images: {class_stats['corrupted_images']}")
            print(f"  - Correct size (224x224x3): {class_stats['correct_size_images']}")
            
        # Convert Counter to regular dict and calculate percentages
        stats["resolution_distribution"] = dict(resolution_counter)
        
        if stats["total_images"] > 0:
            stats["size_compliance"]["percentage_correct"] = (
                stats["size_compliance"]["correct_size"] / stats["total_images"] * 100
            )
        
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Total classes: {stats['total_classes']}")
        print(f"Total images: {stats['total_images']}")
        print(f"Total corrupted: {stats['total_corrupted']}")
        print(f"Images with correct size (224x224x3): {stats['size_compliance']['correct_size']}")
        print(f"Size compliance: {stats['size_compliance']['percentage_correct']:.1f}%")
        
        if stats["corrupted_files"]:
            print(f"\nCorrupted files found: {len(stats['corrupted_files'])}")
            for corrupted in stats["corrupted_files"][:5]:  # Show first 5
                print(f"  - {corrupted['file']} ({corrupted['error'][:50]}...)")
            if len(stats["corrupted_files"]) > 5:
                print(f"  ... and {len(stats['corrupted_files']) - 5} more")
        
        print("\nResolution distribution:")
        for resolution, count in sorted(stats["resolution_distribution"].items()):
            percentage = (count / stats["total_images"]) * 100
            print(f"  - {resolution}: {count} images ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error during verification: {e}")
        stats["error"] = str(e)
    
    return stats


def save_stats(stats: Dict[str, Any], output_path: str = "reports/dataset_stats.json") -> None:
    """Save statistics to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to: {output_file.absolute()}")


def main():
    """Main function to run dataset verification."""
    # Get data root from command line or use default
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    
    print("Animal Classification Dataset Verification")
    print("=" * 60)
    
    # Verify dataset
    stats = verify_dataset(data_root)
    
    # Save statistics
    save_stats(stats)
    
    # Return exit code based on verification results
    if stats.get("error"):
        print(f"\nVerification failed: {stats['error']}")
        sys.exit(1)
    elif stats["total_corrupted"] > 0:
        print(f"\nWarning: {stats['total_corrupted']} corrupted files found!")
        sys.exit(2)
    else:
        print("\nDataset verification completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
