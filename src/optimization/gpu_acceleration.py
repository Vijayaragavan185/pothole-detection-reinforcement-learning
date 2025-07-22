#!/usr/bin/env python3
"""
⚡ GPU ACCELERATION & MODEL OPTIMIZATION ⚡
Day 8: Advanced performance optimization for production deployment
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.jit import script, trace
import time
import numpy as np
from pathlib import Path
import sys
import psutil
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.agents.advanced_dqn import AdvancedDQNAgent
from src.detectors.real_cnn_detector import create_cnn_detector

class OptimizedPotholeSystem:
    """
    🚀 PRODUCTION-OPTIMIZED POTHOLE DETECTION SYSTEM
    Advanced GPU acceleration, model optimization, and performance tuning
    """
    
    def __init__(self, model_path, optimization_level="balanced"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimization_level = optimization_level  # "speed", "balanced", "accuracy"
        
        print(f"⚡ Initializing OPTIMIZED Pothole Detection System")
        print(f"   🎯 Device: {self.device}")
        print(f"   🔧 Optimization Level: {optimization_level}")
        
        # Load and optimize models
        self.load_and_optimize_models(model_path)
        
        # Performance metrics
        self.optimization_stats = {}
        
    def load_and_optimize_models(self, model_path):
        """Load and apply comprehensive optimizations"""
        
        # 1. Load RL Agent
        print("🧠 Loading and optimizing RL Agent...")
        self.rl_agent = AdvancedDQNAgent(
            use_double_dqn=True,
            use_dueling=True,
            use_prioritized_replay=True
        )
        
        if Path(model_path).exists():
            self.rl_agent.load_model(model_path)
            print("✅ RL model loaded successfully")
        else:
            print("⚠️ Using randomly initialized RL model")
        
        # 2. Optimize RL Network
        self.optimize_rl_network()
        
        # 3. Load CNN Detector
        print("👁️ Loading and optimizing CNN Detector...")
        self.cnn_detector = create_cnn_detector(use_real_cnn=False)
        
        # 4. Apply memory optimizations
        self.apply_memory_optimizations()
        
    def optimize_rl_network(self):
        """Apply comprehensive RL network optimizations"""
        
        # Move to GPU
        self.rl_agent.q_network = self.rl_agent.q_network.to(self.device)
        self.rl_agent.target_network = self.rl_agent.target_network.to(self.device)
        
        # Set to evaluation mode
        self.rl_agent.q_network.eval()
        self.rl_agent.target_network.eval()
        
        # Apply optimization based on level
        if self.optimization_level == "speed":
            self.apply_speed_optimizations()
        elif self.optimization_level == "balanced":
            self.apply_balanced_optimizations()
        else:  # accuracy
            self.apply_accuracy_optimizations()
    
    def apply_speed_optimizations(self):
        """Apply aggressive speed optimizations"""
        print("🚀 Applying SPEED optimizations...")
        
        # 1. JIT Compilation
        try:
            dummy_input = torch.randn(1, 5, 224, 224, 3).to(self.device)
            self.rl_agent.q_network = trace(self.rl_agent.q_network, dummy_input)
            print("✅ JIT compilation successful")
        except Exception as e:
            print(f"⚠️ JIT compilation failed: {e}")
        
        # 2. Mixed Precision (if GPU available)
        if self.device.type == "cuda":
            self.enable_mixed_precision()
        
        # 3. CPU Quantization (if on CPU)
        if self.device.type == "cpu":
            self.apply_cpu_quantization()
    
    def apply_balanced_optimizations(self):
        """Apply balanced speed/accuracy optimizations"""
        print("⚖️ Applying BALANCED optimizations...")
        
        # 1. Selective JIT compilation
        try:
            # Only compile main network, keep target network flexible
            dummy_input = torch.randn(1, 5, 224, 224, 3).to(self.device)
            self.rl_agent.q_network = script(self.rl_agent.q_network)
            print("✅ Selective JIT compilation successful")
        except Exception as e:
            print(f"⚠️ JIT compilation failed: {e}")
        
        # 2. Memory optimization
        torch.backends.cudnn.benchmark = True
        
        # 3. Gradient computation disabled
        for param in self.rl_agent.q_network.parameters():
            param.requires_grad = False
    
    def apply_accuracy_optimizations(self):
        """Apply accuracy-preserving optimizations"""
        print("🎯 Applying ACCURACY optimizations...")
        
        # 1. Keep models in full precision
        # 2. Enable deterministic operations
        if self.device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # 3. Memory pinning for faster transfers
        if self.device.type == "cuda":
            torch.multiprocessing.set_sharing_strategy('file_system')
    
    def enable_mixed_precision(self):
        """Enable mixed precision for GPU acceleration"""
        try:
            # Check if GPU supports mixed precision
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
                self.use_amp = True
                self.scaler = torch.cuda.amp.GradScaler()
                print("✅ Mixed precision enabled")
            else:
                self.use_amp = False
                print("⚠️ GPU doesn't support mixed precision")
        except:
            self.use_amp = False
    
    def apply_cpu_quantization(self):
        """Apply dynamic quantization for CPU inference"""
        try:
            # Dynamic quantization for CPU
            self.rl_agent.q_network = quantization.quantize_dynamic(
                self.rl_agent.q_network,
                {nn.Linear},
                dtype=torch.qint8
            )
            print("✅ CPU quantization applied")
        except Exception as e:
            print(f"⚠️ CPU quantization failed: {e}")
    
    def apply_memory_optimizations(self):
        """Apply memory usage optimizations"""
        print("💾 Applying memory optimizations...")
        
        # 1. Clear unnecessary caches
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # 2. Set memory allocation strategy
        if self.device.type == "cuda":
            # Use memory pool for efficiency
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        # 3. Enable memory mapping for large models
        torch.serialization.add_safe_globals(['numpy', 'collections'])
        
        print("✅ Memory optimizations applied")
    
    def benchmark_performance(self, iterations=1000):
        """Comprehensive performance benchmarking"""
        print(f"📊 Running performance benchmark ({iterations} iterations)...")
        
        # Prepare test data
        test_sequences = [
            torch.randn(1, 5, 224, 224, 3).to(self.device) 
            for _ in range(iterations)
        ]
        
        # Warmup
        print("🔥 Warming up...")
        for _ in range(50):
            with torch.no_grad():
                _ = self.rl_agent.q_network(test_sequences[0])
        
        # Synchronize if GPU
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark RL Agent
        print("🤖 Benchmarking RL Agent...")
        start_time = time.time()
        
        for test_seq in test_sequences:
            with torch.no_grad():
                if hasattr(self, 'use_amp') and self.use_amp:
                    with torch.cuda.amp.autocast():
                        _ = self.rl_agent.q_network(test_seq)
                else:
                    _ = self.rl_agent.q_network(test_seq)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_inference_time = total_time / iterations
        fps = 1.0 / avg_inference_time
        
        # Memory usage
        memory_used = self.get_memory_usage()
        
        # Store results
        self.optimization_stats = {
            'optimization_level': self.optimization_level,
            'device': str(self.device),
            'total_time': total_time,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'fps': fps,
            'memory_usage_mb': memory_used,
            'mixed_precision': getattr(self, 'use_amp', False),
            'jit_compiled': hasattr(self.rl_agent.q_network, '_c'),
            'iterations': iterations
        }
        
        # Display results
        print(f"🎉 OPTIMIZATION RESULTS:")
        print(f"   ⚡ FPS: {fps:.1f}")
        print(f"   🔄 Avg inference time: {avg_inference_time * 1000:.2f}ms")
        print(f"   💾 Memory usage: {memory_used:.1f}MB")
        print(f"   🎯 Device: {self.device}")
        print(f"   🚀 Mixed precision: {getattr(self, 'use_amp', False)}")
        
        return self.optimization_stats
    
    def get_memory_usage(self):
        """Get current memory usage"""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
    
    def optimize_for_deployment(self, target_fps=30):
        """Optimize system for specific deployment requirements"""
        print(f"🎯 Optimizing for deployment (target: {target_fps} FPS)...")
        
        # Benchmark current performance
        current_stats = self.benchmark_performance(100)
        current_fps = current_stats['fps']
        
        print(f"📊 Current FPS: {current_fps:.1f}")
        
        if current_fps >= target_fps:
            print(f"✅ Already meeting target FPS!")
            return True
        
        # Try progressively more aggressive optimizations
        optimization_strategies = [
            ("balanced", "⚖️ Trying balanced optimizations..."),
            ("speed", "🚀 Trying aggressive speed optimizations...")
        ]
        
        for strategy, message in optimization_strategies:
            print(message)
            
            # Reload and re-optimize
            self.optimization_level = strategy
            self.optimize_rl_network()
            
            # Test performance
            test_stats = self.benchmark_performance(50)
            test_fps = test_stats['fps']
            
            print(f"📈 {strategy.title()} optimization FPS: {test_fps:.1f}")
            
            if test_fps >= target_fps:
                print(f"✅ Target FPS achieved with {strategy} optimization!")
                self.optimization_stats = test_stats
                return True
        
        print(f"⚠️ Could not achieve target FPS. Current: {test_fps:.1f}")
        return False
    
    def save_optimized_model(self, output_path):
        """Save optimized model for deployment"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save optimized RL model
        rl_model_path = output_path / "optimized_rl_model.pth"
        
        if hasattr(self.rl_agent.q_network, '_c'):  # JIT compiled
            self.rl_agent.q_network.save(str(rl_model_path))
        else:
            torch.save(self.rl_agent.q_network.state_dict(), rl_model_path)
        
        # Save optimization metadata
        metadata = {
            'optimization_level': self.optimization_level,
            'device': str(self.device),
            'performance_stats': self.optimization_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = output_path / "optimization_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Optimized model saved to: {output_path}")
        print(f"   📊 Performance: {self.optimization_stats['fps']:.1f} FPS")
        print(f"   💾 Memory: {self.optimization_stats['memory_usage_mb']:.1f} MB")


class BatchProcessor:
    """
    📦 BATCH PROCESSING OPTIMIZATION
    Process multiple video sequences efficiently
    """
    
    def __init__(self, optimized_system, batch_size=8):
        self.system = optimized_system
        self.batch_size = batch_size
        
    def process_batch(self, sequences_batch):
        """Process multiple sequences in parallel"""
        # Stack sequences into batch tensor
        if len(sequences_batch) != self.batch_size:
            # Pad batch if necessary
            while len(sequences_batch) < self.batch_size:
                sequences_batch.append(sequences_batch[-1])  # Repeat last sequence
        
        # Convert to batch tensor
        batch_tensor = torch.stack([
            torch.FloatTensor(seq) for seq in sequences_batch
        ]).to(self.system.device)
        
        # Process batch
        with torch.no_grad():
            batch_outputs = self.system.rl_agent.q_network(batch_tensor)
            
        # Extract individual results
        results = []
        for i in range(len(sequences_batch)):
            action = torch.argmax(batch_outputs[i]).item()
            confidence = torch.softmax(batch_outputs[i], dim=0).max().item()
            
            results.append({
                'action': action,
                'confidence': confidence,
                'threshold': self.system.rl_agent.action_thresholds[action] if hasattr(self.system.rl_agent, 'action_thresholds') else [0.3, 0.5, 0.7, 0.8, 0.9][action]
            })
        
        return results


# Performance testing and comparison
def run_optimization_comparison():
    """Run comprehensive optimization comparison"""
    
    print("🔥" * 60)
    print("DAY 8: PERFORMANCE OPTIMIZATION COMPARISON")
    print("🔥" * 60)
    
    model_path = "results/ultimate_comparison/models/best_ultimate_dqn_model.pth"
    
    # Test different optimization levels
    optimization_levels = ["accuracy", "balanced", "speed"]
    results = {}
    
    for level in optimization_levels:
        print(f"\n🧪 Testing {level.upper()} optimization...")
        
        try:
            # Create optimized system
            opt_system = OptimizedPotholeSystem(model_path, optimization_level=level)
            
            # Benchmark performance
            stats = opt_system.benchmark_performance(500)
            results[level] = stats
            
            print(f"✅ {level.title()} optimization complete!")
            
        except Exception as e:
            print(f"❌ Error with {level} optimization: {e}")
            results[level] = None
    
    # Compare results
    print(f"\n📊 OPTIMIZATION COMPARISON RESULTS:")
    print("="*60)
    
    for level, stats in results.items():
        if stats:
            print(f"{level.upper()} Optimization:")
            print(f"   ⚡ FPS: {stats['fps']:.1f}")
            print(f"   🔄 Latency: {stats['avg_inference_time_ms']:.2f}ms")
            print(f"   💾 Memory: {stats['memory_usage_mb']:.1f}MB")
            print()
    
    # Save comparison results
    results_path = Path("results/optimizations/day8_optimization_comparison.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved: {results_path}")
    
    return results


if __name__ == "__main__":
    print("⚡ STARTING DAY 8 PERFORMANCE OPTIMIZATION!")
    print("="*60)
    
    # Run optimization comparison
    comparison_results = run_optimization_comparison()
    
    print("🎉 DAY 8 OPTIMIZATION COMPLETED!")
    print("🚀 System optimized for production deployment!")
