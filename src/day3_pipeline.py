#!/usr/bin/env python3
"""
Day 3 Pipeline: Advanced Preprocessing and Data Augmentation
Integrates all Day 3 components for complete preprocessing workflow
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.mask_processor import MaskProcessor
from src.utils.data_augmentation import VideoAugmentation
from src.utils.quality_enhancer import VideoQualityEnhancer
from src.utils.sequence_optimizer import SequenceOptimizer
import time

def run_day3_pipeline():
    """Execute complete Day 3 preprocessing pipeline"""
    
    print("="*60)
    print("DAY 3 PIPELINE: ADVANCED PREPROCESSING & DATA AUGMENTATION")
    print("="*60)
    
    start_time = time.time()
    
    # Step 1: Process Ground Truth Masks
    print("\n🎯 STEP 1: Processing Ground Truth Masks...")
    mask_processor = MaskProcessor()
    mask_success, mask_failed, mask_sequences = mask_processor.process_all_splits()
    
    # Step 2: Quality Enhancement (Training data only)
    print("\n🎯 STEP 2: Enhancing Video Quality...")
    quality_enhancer = VideoQualityEnhancer()
    enhanced_count = quality_enhancer.enhance_processed_data("train", enhancement_level="medium")
    
    # Step 3: Data Augmentation (Training data only)
    print("\n🎯 STEP 3: Creating Augmented Dataset...")
    augmenter = VideoAugmentation()
    train_aug_count = augmenter.create_augmented_dataset("train", augmentation_factor=2)
    val_aug_count = augmenter.create_augmented_dataset("val", augmentation_factor=1)
    
    # Step 4: Sequence Optimization
    print("\n🎯 STEP 4: Optimizing Sequences...")
    optimizer = SequenceOptimizer()
    optimization_stats = optimizer.generate_optimization_report()
    
    # Generate final report
    end_time = time.time()
    pipeline_duration = end_time - start_time
    
    print("\n" + "="*60)
    print("DAY 3 PIPELINE COMPLETION REPORT")
    print("="*60)
    
    print(f"\n📊 Processing Statistics:")
    print(f"  Mask Processing:")
    print(f"    ✅ Successful: {mask_success}")
    print(f"    ❌ Failed: {mask_failed}")
    print(f"    📁 Total mask sequences: {mask_sequences}")
    
    print(f"\n  Quality Enhancement:")
    print(f"    ✅ Enhanced sequences: {enhanced_count}")
    
    print(f"\n  Data Augmentation:")
    print(f"    ✅ Train augmented: {train_aug_count}")
    print(f"    ✅ Val augmented: {val_aug_count}")
    
    print(f"\n  Sequence Optimization:")
    for split, count in optimization_stats.items():
        print(f"    ✅ {split}: {count} optimized sequences")
    
    print(f"\n⏱️  Pipeline Duration: {pipeline_duration:.2f} seconds")
    
    # Success metrics
    success_rate = mask_success / (mask_success + mask_failed) * 100 if (mask_success + mask_failed) > 0 else 0
    
    print(f"\n🎯 Success Metrics:")
    print(f"  Overall success rate: {success_rate:.1f}%")
    print(f"  Total processed sequences: {sum(optimization_stats.values())}")
    
    if success_rate >= 90:
        print(f"  ✅ Day 3 COMPLETED SUCCESSFULLY!")
    elif success_rate >= 70:
        print(f"  ⚠️  Day 3 completed with some issues")
    else:
        print(f"  ❌ Day 3 has significant issues - review logs")
    
    return {
        "mask_processing": {"success": mask_success, "failed": mask_failed, "sequences": mask_sequences},
        "quality_enhancement": {"enhanced": enhanced_count},
        "data_augmentation": {"train": train_aug_count, "val": val_aug_count},
        "sequence_optimization": optimization_stats,
        "success_rate": success_rate,
        "duration": pipeline_duration
    }

if __name__ == "__main__":
    results = run_day3_pipeline()
    
    print(f"\n🚀 Ready for Day 4: RL Environment Implementation")
    print(f"📁 Preprocessed data available in processed_frames directories")
