#!/usr/bin/env python3
"""
⚡ PERFORMANCE BENCHMARKING SYSTEM ⚡
Comprehensive benchmarking of the integrated CNN+RL system
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import psutil
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.environment.pothole_env import VideoBasedPotholeEnv
from src.agents.advanced_dqn import AdvancedDQNAgent
from src.detectors.real_cnn_detector import create_cnn_detector
from src.demo.live_pothole_processor import LivePotholeProcessor

class PerformanceBenchmarks:
    """
    📊 COMPREHENSIVE PERFORMANCE BENCHMARKING SUITE
    Tests all components of the integrated system
    """
    
    def __init__(self):
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create results directory
        self.results_dir = Path("results/benchmarks")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("⚡ Performance Benchmarking Suite Initialized!")
        print(f"   🎯 Device: {self.device}")
        print(f"   💾 Results directory: {self.results_dir}")
    
    def benchmark_cnn_detector(self, iterations=1000):
        """🧠 Benchmark CNN detector performance"""
        print(f"\n🧠 BENCHMARKING CNN DETECTOR ({iterations} iterations)")
        print("="*50)
        
        # Test both real and simulated CNN
        for use_real_cnn in [False, True]:  # Start with simulated for baseline
            detector_type = "Real CNN" if use_real_cnn else "Simulated CNN"
            print(f"\n🔬 Testing {detector_type}...")
            
            try:
                # Create detector
                detector = create_cnn_detector(use_real_cnn=use_real_cnn)
                
                # Prepare test data
                test_sequences = [
                    np.random.rand(5, 224, 224, 3).astype(np.float32) 
                    for _ in range(iterations)
                ]
                
                # Benchmark detection
                detection_times = []
                confidences = []
                memory_usage = []
                
                # Warmup
                for _ in range(10):
                    _ = detector.detect_pothole_confidence(test_sequences[0])
                
                # Actual benchmark
                start_time = time.time()
                
                for i, sequence in enumerate(test_sequences):
                    iter_start = time.time()
                    
                    confidence = detector.detect_pothole_confidence(sequence)
                    
                    iter_time = time.time() - iter_start
                    detection_times.append(iter_time)
                    confidences.append(confidence)
                    
                    # Track memory every 100 iterations
                    if i % 100 == 0:
                        memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)
                
                total_time = time.time() - start_time
                
                # Calculate statistics
                stats = {
                    'detector_type': detector_type,
                    'total_time': total_time,
                    'avg_detection_time': np.mean(detection_times),
                    'min_detection_time': np.min(detection_times),
                    'max_detection_time': np.max(detection_times),
                    'std_detection_time': np.std(detection_times),
                    'avg_fps': 1.0 / np.mean(detection_times),
                    'avg_confidence': np.mean(confidences),
                    'confidence_std': np.std(confidences),
                    'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
                    'iterations': iterations
                }
                
                self.results[f'cnn_{detector_type.lower().replace(" ", "_")}'] = stats
                
                # Display results
                print(f"✅ {detector_type} Results:")
                print(f"   ⚡ Average FPS: {stats['avg_fps']:.1f}")
                print(f"   🔄 Avg detection time: {stats['avg_detection_time']*1000:.2f}ms")
                print(f"   📊 Min/Max time: {stats['min_detection_time']*1000:.2f}ms / {stats['max_detection_time']*1000:.2f}ms")
                print(f"   🎯 Avg confidence: {stats['avg_confidence']:.3f} ± {stats['confidence_std']:.3f}")
                print(f"   💾 Memory usage: {stats['avg_memory_mb']:.1f}MB")
                
            except Exception as e:
                print(f"❌ Error benchmarking {detector_type}: {e}")
    
    def benchmark_rl_agent(self, model_path=None, iterations=1000):
        """🤖 Benchmark RL agent performance"""
        print(f"\n🤖 BENCHMARKING RL AGENT ({iterations} iterations)")
        print("="*50)
        
        # Test different RL configurations
        configs = [
            {"name": "Standard DQN", "use_double_dqn": False, "use_dueling": False, "use_prioritized_replay": False},
            {"name": "Dueling DQN", "use_double_dqn": False, "use_dueling": True, "use_prioritized_replay": False},
            {"name": "Ultimate DQN", "use_double_dqn": True, "use_dueling": True, "use_prioritized_replay": True}
        ]
        
        for config in configs:
            print(f"\n🔬 Testing {config['name']}...")
            
            try:
                # Create agent
                agent = AdvancedDQNAgent(
                    use_double_dqn=config["use_double_dqn"],
                    use_dueling=config["use_dueling"],
                    use_prioritized_replay=config["use_prioritized_replay"]
                )
                
                # Load model if provided
                if model_path and Path(model_path).exists():
                    try:
                        agent.load_model(model_path)
                        print(f"   ✅ Loaded model from {model_path}")
                    except:
                        print(f"   ⚠️ Could not load model, using random weights")
                
                agent.q_network.eval()
                
                # Prepare test data
                test_sequences = [
                    np.random.rand(5, 224, 224, 3).astype(np.float32)
                    for _ in range(iterations)
                ]
                
                # Benchmark inference
                inference_times = []
                actions = []
                
                # Warmup
                for _ in range(10):
                    _ = agent.act(test_sequences[0], training=False)
                
                # Actual benchmark
                start_time = time.time()
                
                for sequence in test_sequences:
                    iter_start = time.time()
                    
                    action = agent.act(sequence, training=False)
                    
                    iter_time = time.time() - iter_start
                    inference_times.append(iter_time)
                    actions.append(action)
                
                total_time = time.time() - start_time
                
                # Calculate statistics
                stats = {
                    'agent_type': config['name'],
                    'total_time': total_time,
                    'avg_inference_time': np.mean(inference_times),
                    'min_inference_time': np.min(inference_times),
                    'max_inference_time': np.max(inference_times),
                    'std_inference_time': np.std(inference_times),
                    'avg_fps': 1.0 / np.mean(inference_times),
                    'action_distribution': np.bincount(actions, minlength=5).tolist(),
                    'model_parameters': sum(p.numel() for p in agent.q_network.parameters()),
                    'iterations': iterations
                }
                
                self.results[f'rl_{config["name"].lower().replace(" ", "_")}'] = stats
                
                # Display results
                print(f"✅ {config['name']} Results:")
                print(f"   ⚡ Average FPS: {stats['avg_fps']:.1f}")
                print(f"   🔄 Avg inference time: {stats['avg_inference_time']*1000:.2f}ms")
                print(f"   📊 Min/Max time: {stats['min_inference_time']*1000:.2f}ms / {stats['max_inference_time']*1000:.2f}ms")
                print(f"   🧠 Model parameters: {stats['model_parameters']:,}")
                print(f"   🎯 Action distribution: {stats['action_distribution']}")
                
            except Exception as e:
                print(f"❌ Error benchmarking {config['name']}: {e}")
    
    def benchmark_integrated_system(self, rl_model_path=None, iterations=500):
        """🚀 Benchmark complete integrated system"""
        print(f"\n🚀 BENCHMARKING INTEGRATED SYSTEM ({iterations} iterations)")
        print("="*50)
        
        try:
            # Create integrated system
            processor = LivePotholeProcessor(
                rl_model_path=rl_model_path,
                use_real_cnn=False  # Use simulated for consistent benchmarking
            )
            
            # Prepare test data
            test_frames = [
                np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                for _ in range(iterations)
            ]
            
            # Benchmark end-to-end processing
            processing_times = []
            detection_results = []
            
            # Warmup
            for _ in range(10):
                _, _ = processor.process_single_frame(test_frames[0])
            
            # Actual benchmark
            start_time = time.time()
            
            for frame in test_frames:
                iter_start = time.time()
                
                processed_frame, detection_info = processor.process_single_frame(frame)
                
                iter_time = time.time() - iter_start
                processing_times.append(iter_time)
                detection_results.append(detection_info)
            
            total_time = time.time() - start_time
            
            # Calculate statistics
            detections = sum(1 for d in detection_results if d['pothole_detected'])
            avg_confidence = np.mean([d['confidence'] for d in detection_results])
            
            stats = {
                'system_type': 'Integrated CNN+RL',
                'total_time': total_time,
                'avg_processing_time': np.mean(processing_times),
                'min_processing_time': np.min(processing_times),
                'max_processing_time': np.max(processing_times),
                'std_processing_time': np.std(processing_times),
                'avg_fps': 1.0 / np.mean(processing_times),
                'total_detections': detections,
                'detection_rate': detections / iterations * 100,
                'avg_confidence': avg_confidence,
                'iterations': iterations
            }
            
            self.results['integrated_system'] = stats
            
            # Display results
            print(f"✅ Integrated System Results:")
            print(f"   ⚡ Average FPS: {stats['avg_fps']:.1f}")
            print(f"   🔄 Avg processing time: {stats['avg_processing_time']*1000:.2f}ms")
            print(f"   📊 Min/Max time: {stats['min_processing_time']*1000:.2f}ms / {stats['max_processing_time']*1000:.2f}ms")
            print(f"   🕳️ Total detections: {stats['total_detections']}")
            print(f"   📈 Detection rate: {stats['detection_rate']:.1f}%")
            print(f"   🎯 Average confidence: {stats['avg_confidence']:.3f}")
            
        except Exception as e:
            print(f"❌ Error benchmarking integrated system: {e}")
    
    def benchmark_memory_usage(self):
        """💾 Benchmark memory usage"""
        print(f"\n💾 BENCHMARKING MEMORY USAGE")
        print("="*30)
        
        try:
            # Get baseline memory
            baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Test environment loading
            print("🎮 Testing environment loading...")
            env_start_mem = psutil.Process().memory_info().rss / 1024 / 1024
            env = VideoBasedPotholeEnv(target_sequences=1000, balanced=True, verbose=False)
            env_end_mem = psutil.Process().memory_info().rss / 1024 / 1024
            env.close()
            
            # Test CNN detector
            print("🧠 Testing CNN detector...")
            cnn_start_mem = psutil.Process().memory_info().rss / 1024 / 1024
            detector = create_cnn_detector(use_real_cnn=True)
            cnn_end_mem = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Test RL agent
            print("🤖 Testing RL agent...")
            rl_start_mem = psutil.Process().memory_info().rss / 1024 / 1024
            agent = AdvancedDQNAgent(use_double_dqn=True, use_dueling=True, use_prioritized_replay=True)
            rl_end_mem = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Memory stats
            memory_stats = {
                'baseline_mb': baseline_memory,
                'environment_loading_mb': env_end_mem - env_start_mem,
                'cnn_detector_mb': cnn_end_mem - cnn_start_mem,
                'rl_agent_mb': rl_end_mem - rl_start_mem,
                'total_system_mb': rl_end_mem - baseline_memory
            }
            
            self.results['memory_usage'] = memory_stats
            
            print(f"✅ Memory Usage Results:")
            print(f"   📊 Baseline: {memory_stats['baseline_mb']:.1f}MB")
            print(f"   🎮 Environment: +{memory_stats['environment_loading_mb']:.1f}MB")
            print(f"   🧠 CNN Detector: +{memory_stats['cnn_detector_mb']:.1f}MB")
            print(f"   🤖 RL Agent: +{memory_stats['rl_agent_mb']:.1f}MB")
            print(f"   🔥 Total System: {memory_stats['total_system_mb']:.1f}MB")
            
        except Exception as e:
            print(f"❌ Error benchmarking memory: {e}")
    
    def generate_benchmark_report(self):
        """📊 Generate comprehensive benchmark report"""
        if not self.results:
            print("❌ No benchmark results to report")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        results_path = self.results_dir / f"benchmark_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate visualizations
        self.create_benchmark_visualizations(timestamp)
        
        # Generate text report
        report_path = self.results_dir / f"benchmark_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("⚡ PERFORMANCE BENCHMARK REPORT ⚡\n")
            f.write("="*50 + "\n\n")
            
            for component, stats in self.results.items():
                f.write(f"🔬 {component.upper()}\n")
                f.write("-" * 30 + "\n")
                for key, value in stats.items():
                    if isinstance(value, float):
                        if 'time' in key:
                            f.write(f"{key}: {value*1000:.2f}ms\n")
                        else:
                            f.write(f"{key}: {value:.3f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
        
        print(f"\n📊 BENCHMARK REPORT GENERATED:")
        print(f"   📋 JSON Results: {results_path}")
        print(f"   📄 Text Report: {report_path}")
        print(f"   📈 Visualizations: {self.results_dir}/benchmark_plots_{timestamp}.png")
    
    def create_benchmark_visualizations(self, timestamp):
        """📈 Create benchmark visualizations"""
        if not self.results:
            return
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Benchmark Results', fontsize=16, fontweight='bold')
        
        # FPS comparison
        fps_data = {}
        for key, stats in self.results.items():
            if 'avg_fps' in stats:
                fps_data[key.replace('_', ' ').title()] = stats['avg_fps']
        
        if fps_data:
            axes[0, 0].bar(fps_data.keys(), fps_data.values())
            axes[0, 0].set_title('Average FPS Comparison')
            axes[0, 0].set_ylabel('FPS')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Processing time comparison
        time_data = {}
        for key, stats in self.results.items():
            if 'avg_detection_time' in stats:
                time_data[key.replace('_', ' ').title()] = stats['avg_detection_time'] * 1000
            elif 'avg_inference_time' in stats:
                time_data[key.replace('_', ' ').title()] = stats['avg_inference_time'] * 1000
            elif 'avg_processing_time' in stats:
                time_data[key.replace('_', ' ').title()] = stats['avg_processing_time'] * 1000
        
        if time_data:
            axes[0, 1].bar(time_data.keys(), time_data.values())
            axes[0, 1].set_title('Average Processing Time')
            axes[0, 1].set_ylabel('Time (ms)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage
        if 'memory_usage' in self.results:
            mem_stats = self.results['memory_usage']
            mem_labels = ['Environment', 'CNN Detector', 'RL Agent']
            mem_values = [
                mem_stats['environment_loading_mb'],
                mem_stats['cnn_detector_mb'],
                mem_stats['rl_agent_mb']
            ]
            
            axes[0, 2].pie(mem_values, labels=mem_labels, autopct='%1.1f%%')
            axes[0, 2].set_title('Memory Usage Distribution')
        
        # Performance trends (if we have timing data)
        axes[1, 0].text(0.5, 0.5, 'Performance\nTrends\n(Future)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Performance Trends')
        
        # System comparison
        if 'integrated_system' in self.results:
            system_stats = self.results['integrated_system']
            metrics = ['FPS', 'Detection Rate %', 'Avg Confidence']
            values = [
                system_stats['avg_fps'],
                system_stats['detection_rate'],
                system_stats['avg_confidence'] * 100
            ]
            
            axes[1, 1].bar(metrics, values)
            axes[1, 1].set_title('Integrated System Performance')
        
        # Hardware utilization
        device_info = f"Device: {self.device}\nGPU: {'Available' if torch.cuda.is_available() else 'Not Available'}"
        axes[1, 2].text(0.5, 0.5, device_info, ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Hardware Configuration')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"benchmark_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_benchmark_suite(self, rl_model_path=None):
        """🚀 Run complete benchmark suite"""
        print("🚀" * 60)
        print("ULTIMATE PERFORMANCE BENCHMARK SUITE")
        print("🚀" * 60)
        
        # Run all benchmarks
        self.benchmark_memory_usage()
        self.benchmark_cnn_detector(iterations=500)
        self.benchmark_rl_agent(model_path=rl_model_path, iterations=500)
        self.benchmark_integrated_system(rl_model_path=rl_model_path, iterations=200)
        
        # Generate comprehensive report
        self.generate_benchmark_report()
        
        print("\n🎉 FULL BENCHMARK SUITE COMPLETED!")
        print("📊 Check results/benchmarks/ for detailed reports and visualizations")


# Example usage
if __name__ == "__main__":
    print("⚡ STARTING PERFORMANCE BENCHMARK SUITE!")
    print("="*60)
    
    # Initialize benchmarking system
    benchmark = PerformanceBenchmarks()
    
    # Run full benchmark suite
    benchmark.run_full_benchmark_suite(
        rl_model_path="results/models/best_ultimate_dqn_model.pth"
    )
    
    print("🎉 BENCHMARKING COMPLETED - SYSTEM READY FOR PRODUCTION!")
