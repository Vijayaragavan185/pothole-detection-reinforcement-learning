#!/usr/bin/env python3
"""
Day 3 Pipeline: Advanced Preprocessing and Data Augmentation
Integrates all Day 3 components for complete preprocessing workflow
Updated with correct execution order and comprehensive validation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.mask_processor import MaskProcessor
from src.utils.data_augmentation import VideoAugmentation
from src.utils.sequence_optimizer import SequenceOptimizer
from src.utils.video_analyzer import VideoAnalyzer
from src.utils.data_validator import DatasetValidator
import time
import json
from datetime import datetime

class Day3Pipeline:
    """Complete Day 3 preprocessing pipeline with validation and reporting"""
    
    def __init__(self):
        self.results = {
            "mask_processing": {"success": 0, "failed": 0, "sequences": 0, "skipped": False},
            "sequence_optimization": {"stats": {}, "skipped": False},
            "data_augmentation": {"train": 0, "val": 0, "skipped": False},
            "quality_enhancement": {"enhanced": 0, "skipped": True},
            "video_analysis": {"completed": False, "skipped": False},
            "success_rate": 0.0,
            "duration": 0.0,
            "total_sequences": 0
        }
    
    def check_existing_work(self):
        """Check what work has already been completed to avoid duplication"""
        from configs.config import PATHS
        
        checks = {
            "ground_truth_exists": (PATHS["ground_truth"] / "train").exists(),
            "optimized_exists": (PATHS["processed_frames"] / "train_optimized").exists(),
            "augmented_exists": (PATHS["processed_frames"] / "train_augmented").exists(),
            "enhanced_exists": (PATHS["processed_frames"] / "train_enhanced").exists()
        }
        
        print("🔍 Checking existing work...")
        for check, status in checks.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {check.replace('_', ' ').title()}: {status}")
        
        return checks
    
    def step1_process_ground_truth(self, force_rerun=False):
        """Step 1: Process Ground Truth Masks"""
        print("\n🎯 STEP 1: Processing Ground Truth Masks...")
        
        if not force_rerun:
            from configs.config import PATHS
            if (PATHS["ground_truth"] / "train").exists():
                print("  ✅ Ground truth already processed - skipping")
                self.results["mask_processing"]["skipped"] = True
                # Count existing sequences
                train_gt = PATHS["ground_truth"] / "train"
                if train_gt.exists():
                    total_seqs = sum(len(list((d / "sequences").glob("*.npy"))) 
                                   for d in train_gt.iterdir() if d.is_dir() and (d / "sequences").exists())
                    self.results["mask_processing"]["sequences"] = total_seqs
                    self.results["mask_processing"]["success"] = len([d for d in train_gt.iterdir() if d.is_dir()])
                return
        
        try:
            mask_processor = MaskProcessor()
            mask_success, mask_failed, mask_sequences = mask_processor.process_all_splits()
            
            self.results["mask_processing"]["success"] = mask_success
            self.results["mask_processing"]["failed"] = mask_failed
            self.results["mask_processing"]["sequences"] = mask_sequences
            
        except Exception as e:
            print(f"  ❌ Error in mask processing: {e}")
            self.results["mask_processing"]["failed"] = 999
    
    def step2_sequence_optimization(self, force_rerun=False):
        """Step 2: Sequence Optimization (MOVED UP - Critical Order)"""
        print("\n🎯 STEP 2: Optimizing Sequences...")
        
        if not force_rerun:
            from configs.config import PATHS
            if (PATHS["processed_frames"] / "train_optimized").exists():
                print("  ✅ Sequences already optimized - skipping")
                self.results["sequence_optimization"]["skipped"] = True
                # Count existing optimized sequences
                for split in ["train", "val", "test"]:
                    opt_dir = PATHS["processed_frames"] / f"{split}_optimized"
                    if opt_dir.exists():
                        total_seqs = sum(len(list((d / "sequences").glob("optimized_*.npy"))) 
                                       for d in opt_dir.iterdir() if d.is_dir())
                        self.results["sequence_optimization"]["stats"][split] = total_seqs
                return
        
        try:
            optimizer = SequenceOptimizer()
            optimization_stats = optimizer.generate_optimization_report()
            self.results["sequence_optimization"]["stats"] = optimization_stats
            
        except Exception as e:
            print(f"  ❌ Error in sequence optimization: {e}")
            self.results["sequence_optimization"]["stats"] = {"train": 0, "val": 0, "test": 0}
    
    def step3_data_augmentation(self, force_rerun=False):
        """Step 3: Data Augmentation (Moved after optimization)"""
        print("\n🎯 STEP 3: Creating Augmented Dataset...")
        
        if not force_rerun:
            from configs.config import PATHS
            if (PATHS["processed_frames"] / "train_augmented").exists():
                print("  ✅ Data already augmented - skipping")
                self.results["data_augmentation"]["skipped"] = True
                # Count existing augmented sequences
                for split in ["train", "val"]:
                    aug_dir = PATHS["processed_frames"] / f"{split}_augmented"
                    if aug_dir.exists():
                        total_seqs = sum(len(list((d / "sequences").glob("*aug*.npy"))) 
                                       for d in aug_dir.iterdir() if d.is_dir())
                        self.results["data_augmentation"][split] = total_seqs
                return
        
        try:
            # Check if we have required dependencies
            try:
                import albumentations
            except ImportError:
                print("  ⚠️  Installing albumentations...")
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "albumentations==1.3.1"], 
                             check=True, capture_output=True)
                import albumentations
            
            augmenter = VideoAugmentation()
            train_aug_count = augmenter.create_augmented_dataset("train", augmentation_factor=2)
            val_aug_count = augmenter.create_augmented_dataset("val", augmentation_factor=1)
            
            self.results["data_augmentation"]["train"] = train_aug_count
            self.results["data_augmentation"]["val"] = val_aug_count
            
        except Exception as e:
            print(f"  ❌ Error in data augmentation: {e}")
            self.results["data_augmentation"]["train"] = 0
            self.results["data_augmentation"]["val"] = 0
    
    def step4_quality_enhancement(self, enable_quality=False, level="light"):
        """Step 4: Quality Enhancement (Optional - Performance Heavy)"""
        print("\n🎯 STEP 4: Video Quality Enhancement...")
        
        if not enable_quality:
            print("  ⚠️  Quality enhancement disabled (use --enable-quality to run)")
            print("  ℹ️  Reason: Time-intensive operation (15-30 minutes)")
            self.results["quality_enhancement"]["skipped"] = True
            return
        
        try:
            # Check for scikit-image dependency
            try:
                from skimage import restoration, filters, exposure
            except ImportError:
                print("  ⚠️  Installing scikit-image...")
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "scikit-image"], 
                             check=True, capture_output=True)
            
            from src.utils.quality_enhancer import VideoQualityEnhancer
            quality_enhancer = VideoQualityEnhancer()
            enhanced_count = quality_enhancer.enhance_processed_data("train", enhancement_level=level)
            
            self.results["quality_enhancement"]["enhanced"] = enhanced_count
            self.results["quality_enhancement"]["skipped"] = False
            
        except Exception as e:
            print(f"  ❌ Error in quality enhancement: {e}")
            self.results["quality_enhancement"]["enhanced"] = 0
    
    def step5_video_analysis(self):
        """Step 5: Generate Video Analysis and Reports"""
        print("\n🎯 STEP 5: Generating Analysis Reports...")
        
        try:
            analyzer = VideoAnalyzer()
            analyzer.generate_analysis_report()
            self.results["video_analysis"]["completed"] = True
            
        except Exception as e:
            print(f"  ⚠️  Error in video analysis: {e}")
            print("  ℹ️  Continuing without analysis plots...")
            self.results["video_analysis"]["completed"] = False
    
    def step6_dataset_validation(self):
        """Step 6: Final Dataset Validation"""
        print("\n🎯 STEP 6: Final Dataset Validation...")
        
        try:
            validator = DatasetValidator()
            validation_results = validator.validate_complete_dataset()
            print("  ✅ Dataset validation completed")
            
        except Exception as e:
            print(f"  ⚠️  Error in dataset validation: {e}")
    
    def generate_comprehensive_report(self, pipeline_duration):
        """Generate detailed completion report"""
        print("\n" + "="*70)
        print("DAY 3 PIPELINE COMPREHENSIVE COMPLETION REPORT")
        print("="*70)
        
        print(f"\n📊 Component Processing Statistics:")
        
        # Mask Processing Report
        mask_result = self.results["mask_processing"]
        if mask_result["skipped"]:
            print(f"  🎯 Mask Processing: SKIPPED (Already completed)")
        else:
            print(f"  🎯 Mask Processing:")
        print(f"    ✅ Successful videos: {mask_result['success']}")
        print(f"    ❌ Failed videos: {mask_result['failed']}")
        print(f"    📁 Total mask sequences: {mask_result['sequences']}")
        
        # Sequence Optimization Report
        opt_result = self.results["sequence_optimization"]
        if opt_result["skipped"]:
            print(f"\n  🎯 Sequence Optimization: SKIPPED (Already completed)")
        else:
            print(f"\n  🎯 Sequence Optimization:")
        for split, count in opt_result["stats"].items():
            print(f"    ✅ {split}: {count:,} optimized sequences")
        
        # Data Augmentation Report
        aug_result = self.results["data_augmentation"]
        if aug_result["skipped"]:
            print(f"\n  🎯 Data Augmentation: SKIPPED (Already completed)")
        else:
            print(f"\n  🎯 Data Augmentation:")
        print(f"    ✅ Train augmented: {aug_result['train']:,} sequences")
        print(f"    ✅ Val augmented: {aug_result['val']:,} sequences")
        
        # Quality Enhancement Report
        qual_result = self.results["quality_enhancement"]
        if qual_result["skipped"]:
            print(f"\n  🎯 Quality Enhancement: SKIPPED (Optional/Performance)")
        else:
            print(f"\n  🎯 Quality Enhancement:")
            print(f"    ✅ Enhanced sequences: {qual_result['enhanced']:,}")
        
        # Video Analysis Report
        analysis_result = self.results["video_analysis"]
        status = "✅ COMPLETED" if analysis_result["completed"] else "⚠️ PARTIAL"
        print(f"\n  🎯 Video Analysis: {status}")
        
        # Calculate totals
        total_optimized = sum(opt_result["stats"].values())
        total_augmented = aug_result["train"] + aug_result["val"]
        
        print(f"\n📈 Overall Dataset Statistics:")
        print(f"  📊 Original sequences processed: {mask_result['sequences']:,}")
        print(f"  🎯 Optimized sequences: {total_optimized:,}")
        print(f"  🔄 Augmented sequences: {total_augmented:,}")
        print(f"  📦 Total available sequences: {total_optimized + total_augmented:,}")
        
        # Calculate success metrics
        mask_success_rate = 0
        if mask_result['success'] + mask_result['failed'] > 0:
            mask_success_rate = mask_result['success'] / (mask_result['success'] + mask_result['failed']) * 100
        
        # Performance metrics
        print(f"\n⚡ Performance Metrics:")
        print(f"  ⏱️  Total pipeline duration: {pipeline_duration:.2f} seconds ({pipeline_duration/60:.1f} minutes)")
        print(f"  🎯 Mask processing success rate: {mask_success_rate:.1f}%")
        print(f"  📊 Dataset expansion factor: {((total_optimized + total_augmented) / max(mask_result['sequences'], 1)):.1f}x")
        
        # Success Assessment
        print(f"\n🏆 Day 3 Success Assessment:")
        if mask_success_rate >= 95 and total_optimized > 0:
            print(f"  ✅ DAY 3 COMPLETED WITH EXCELLENCE!")
            print(f"  🎉 Outstanding dataset: {total_optimized + total_augmented:,} sequences ready for RL training")
        elif mask_success_rate >= 80 and total_optimized > 0:
            print(f"  ✅ DAY 3 COMPLETED SUCCESSFULLY!")
            print(f"  🎯 Good dataset: {total_optimized + total_augmented:,} sequences ready for RL training")
        elif total_optimized > 0:
            print(f"  ⚠️  Day 3 completed with some issues")
            print(f"  📊 Usable dataset: {total_optimized + total_augmented:,} sequences available")
        else:
            print(f"  ❌ Day 3 has significant issues - review component outputs")
        
        # Next steps
        print(f"\n🚀 Ready for Next Phase:")
        print(f"  📅 Day 4: RL Environment Implementation")
        print(f"  📁 Preprocessed data location: data/processed_frames/")
        print(f"  🎯 Ground truth location: data/ground_truth/")
        print(f"  💾 Total sequences available for RL training: {total_optimized + total_augmented:,}")
        
        return {
            "total_sequences": total_optimized + total_augmented,
            "success_rate": mask_success_rate,
            "duration": pipeline_duration,
            "components_completed": sum(1 for r in self.results.values() 
                                      if isinstance(r, dict) and not r.get("skipped", False))
        }
    
    def save_pipeline_results(self, final_stats):
        """Save pipeline results for future reference"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path("results") / f"day3_pipeline_results_{timestamp}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        complete_results = {
            "pipeline_execution": {
                "timestamp": timestamp,
                "duration_seconds": final_stats["duration"],
                "success_rate": final_stats["success_rate"],
                "total_sequences": final_stats["total_sequences"]
            },
            "component_results": self.results,
            "final_statistics": final_stats
        }
        
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        print(f"\n💾 Pipeline results saved to: {results_file}")


def run_day3_pipeline(enable_quality=False, quality_level="light", force_rerun=False):
    """Execute complete Day 3 preprocessing pipeline with intelligent skipping"""
    
    print("="*70)
    print("DAY 3 PIPELINE: ADVANCED PREPROCESSING & DATA AUGMENTATION")
    print("Enhanced with validation, error handling, and intelligent skipping")
    print("="*70)
    
    pipeline = Day3Pipeline()
    start_time = time.time()
    
    # Check existing work
    existing_work = pipeline.check_existing_work()
    
    # Execute pipeline steps in correct order
    pipeline.step1_process_ground_truth(force_rerun)
    pipeline.step2_sequence_optimization(force_rerun)  # Moved up
    pipeline.step3_data_augmentation(force_rerun)      # Moved down
    pipeline.step4_quality_enhancement(enable_quality, quality_level)
    pipeline.step5_video_analysis()
    pipeline.step6_dataset_validation()
    
    # Generate comprehensive report
    end_time = time.time()
    pipeline_duration = end_time - start_time
    
    final_stats = pipeline.generate_comprehensive_report(pipeline_duration)
    pipeline.save_pipeline_results(final_stats)
    
    return final_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Day 3 Pipeline: Advanced Preprocessing & Data Augmentation")
    parser.add_argument("--enable-quality", action="store_true", 
                       help="Enable quality enhancement (time-intensive)")
    parser.add_argument("--quality-level", choices=["light", "medium", "heavy"], 
                       default="light", help="Quality enhancement level")
    parser.add_argument("--force-rerun", action="store_true", 
                       help="Force rerun all components (ignore existing work)")
    
    args = parser.parse_args()
    
    results = run_day3_pipeline(
        enable_quality=args.enable_quality,
        quality_level=args.quality_level,
        force_rerun=args.force_rerun
    )
    
    print(f"\n🎯 Day 3 Pipeline Complete!")
    print(f"📊 Total sequences available: {results['total_sequences']:,}")
    print(f"🚀 Ready for Day 4: RL Environment Implementation")
