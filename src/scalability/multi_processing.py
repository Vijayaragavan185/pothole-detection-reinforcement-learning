#!/usr/bin/env python3
"""
🌐 SCALABILITY & MULTI-PROCESSING SYSTEM 🌐
Day 8: Multi-GPU, distributed processing, and cloud deployment preparation
"""

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
import time
import json
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import queue
import threading
from dataclasses import dataclass
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.optimization.gpu_acceleration import OptimizedPotholeSystem

@dataclass
class ProcessingJob:
    """Data structure for processing jobs"""
    job_id: str
    sequences: List[np.ndarray]
    metadata: Dict[str, Any]
    priority: int = 0

class MultiGPUProcessor:
    """
    🚀 MULTI-GPU POTHOLE DETECTION PROCESSOR
    Distribute processing across multiple GPUs for maximum throughput
    """
    
    def __init__(self, model_path, num_gpus=None):
        self.model_path = model_path
        
        # Detect available GPUs
        if torch.cuda.is_available():
            self.num_gpus = num_gpus or torch.cuda.device_count()
            self.device_ids = list(range(self.num_gpus))
            print(f"🚀 Multi-GPU setup: {self.num_gpus} GPUs available")
            print(f"   GPUs: {[torch.cuda.get_device_name(i) for i in self.device_ids]}")
        else:
            self.num_gpus = 1
            self.device_ids = ['cpu']
            print("💻 Using CPU for multi-processing")
        
        # Initialize systems on each device
        self.systems = {}
        self.initialize_systems()
    
    def initialize_systems(self):
        """Initialize optimized systems on each device"""
        for i, device_id in enumerate(self.device_ids):
            print(f"🔧 Initializing system on device {device_id}...")
            
            if device_id == 'cpu':
                device = torch.device('cpu')
            else:
                device = torch.device(f'cuda:{device_id}')
            
            # Create optimized system for this device
            system = OptimizedPotholeSystem(
                self.model_path, 
                optimization_level="speed"
            )
            system.device = device
            
            # Move models to specific device
            system.rl_agent.q_network = system.rl_agent.q_network.to(device)
            
            self.systems[device_id] = system
        
        print(f"✅ All {len(self.systems)} systems initialized!")
    
    def process_parallel(self, job_batches):
        """Process multiple job batches in parallel across GPUs"""
        print(f"🔄 Processing {len(job_batches)} batches across {self.num_gpus} devices...")
        
        if self.num_gpus == 1:
            # Single device processing
            return self._process_single_device(job_batches)
        else:
            # Multi-device processing
            return self._process_multi_device(job_batches)
    
    def _process_single_device(self, job_batches):
        """Process jobs on single device"""
        device_id = self.device_ids[0]
        system = self.systems[device_id]
        
        results = []
        start_time = time.time()
        
        for i, batch in enumerate(job_batches):
            batch_results = []
            
            for job in batch:
                for sequence in job.sequences:
                    # Convert to tensor and move to device
                    seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(system.device)
                    
                    # Process sequence
                    with torch.no_grad():
                        output = system.rl_agent.q_network(seq_tensor)
                        action = torch.argmax(output).item()
                        confidence = torch.softmax(output, dim=1).max().item()
                    
                    batch_results.append({
                        'job_id': job.job_id,
                        'action': action,
                        'confidence': confidence,
                        'device': str(system.device)
                    })
            
            results.extend(batch_results)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                fps = (i + 1) * len(batch) / elapsed
                print(f"   Processed {i+1}/{len(job_batches)} batches, {fps:.1f} jobs/sec")
        
        total_time = time.time() - start_time
        total_jobs = sum(len(batch) for batch in job_batches)
        
        print(f"✅ Single-device processing complete!")
        print(f"   📊 Total jobs: {total_jobs}")
        print(f"   ⏱️ Total time: {total_time:.2f}s")
        print(f"   ⚡ Throughput: {total_jobs/total_time:.1f} jobs/sec")
        
        return results
    
    def _process_multi_device(self, job_batches):
        """Process jobs across multiple devices"""
        
        # Distribute batches across devices
        device_batches = self._distribute_batches(job_batches)
        
        # Create processing threads for each device
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = []
            
            for device_id, batches in device_batches.items():
                if batches:  # Only submit if there are batches to process
                    future = executor.submit(self._process_device_batches, device_id, batches)
                    futures.append(future)
            
            # Collect results from all devices
            all_results = []
            for future in futures:
                device_results = future.result()
                all_results.extend(device_results)
        
        return all_results
    
    def _distribute_batches(self, job_batches):
        """Distribute job batches across available devices"""
        device_batches = {device_id: [] for device_id in self.device_ids}
        
        # Round-robin distribution
        for i, batch in enumerate(job_batches):
            device_id = self.device_ids[i % len(self.device_ids)]
            device_batches[device_id].append(batch)
        
        print(f"📊 Batch distribution:")
        for device_id, batches in device_batches.items():
            print(f"   Device {device_id}: {len(batches)} batches")
        
        return device_batches
    
    def _process_device_batches(self, device_id, batches):
        """Process batches on specific device"""
        system = self.systems[device_id]
        results = []
        
        print(f"🔄 Device {device_id}: Processing {len(batches)} batches...")
        
        start_time = time.time()
        
        for batch in batches:
            for job in batch:
                for sequence in job.sequences:
                    # Convert to tensor and move to device
                    seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(system.device)
                    
                    # Process sequence
                    with torch.no_grad():
                        output = system.rl_agent.q_network(seq_tensor)
                        action = torch.argmax(output).item()
                        confidence = torch.softmax(output, dim=1).max().item()
                    
                    results.append({
                        'job_id': job.job_id,
                        'action': action,
                        'confidence': confidence,
                        'device': str(system.device)
                    })
        
        elapsed = time.time() - start_time
        total_jobs = sum(len(batch) for batch in batches)
        
        print(f"✅ Device {device_id}: {total_jobs} jobs in {elapsed:.2f}s ({total_jobs/elapsed:.1f} jobs/sec)")
        
        return results


class DistributedProcessor:
    """
    🌐 DISTRIBUTED PROCESSING FOR CLOUD DEPLOYMENT
    Scale across multiple machines/containers
    """
    
    def __init__(self, model_path, nodes=1):
        self.model_path = model_path
        self.nodes = nodes
        self.job_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        print(f"🌐 Distributed Processor initialized for {nodes} nodes")
    
    def setup_distributed_training(self, rank, world_size):
        """Setup distributed processing"""
        # This would be used for actual distributed deployment
        torch.distributed.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )
        
        # Wrap model with DistributedDataParallel
        system = OptimizedPotholeSystem(self.model_path)
        system.rl_agent.q_network = DistributedDataParallel(
            system.rl_agent.q_network,
            device_ids=[rank]
        )
        
        return system
    
    def create_processing_cluster(self, num_workers=4):
        """Create cluster of processing workers"""
        print(f"🏗️ Creating processing cluster with {num_workers} workers...")
        
        workers = []
        for i in range(num_workers):
            worker = ProcessingWorker(
                worker_id=i,
                model_path=self.model_path,
                job_queue=self.job_queue,
                result_queue=self.result_queue
            )
            workers.append(worker)
        
        return workers


class ProcessingWorker:
    """
    👷 INDIVIDUAL PROCESSING WORKER
    Handles jobs from queue
    """
    
    def __init__(self, worker_id, model_path, job_queue, result_queue):
        self.worker_id = worker_id
        self.model_path = model_path
        self.job_queue = job_queue
        self.result_queue = result_queue
        
        # Initialize system for this worker
        self.system = OptimizedPotholeSystem(model_path, optimization_level="speed")
        
        print(f"👷 Worker {worker_id} initialized")
    
    def run(self):
        """Main worker processing loop"""
        print(f"🔄 Worker {self.worker_id} starting...")
        
        while True:
            try:
                # Get job from queue (timeout after 5 seconds)
                job = self.job_queue.get(timeout=5)
                
                if job is None:  # Poison pill to stop worker
                    break
                
                # Process job
                results = self.process_job(job)
                
                # Put results in result queue
                self.result_queue.put(results)
                
                # Mark job as done
                self.job_queue.task_done()
                
            except queue.Empty:
                # No job available, continue waiting
                continue
            except Exception as e:
                print(f"❌ Worker {self.worker_id} error: {e}")
                continue
        
        print(f"✅ Worker {self.worker_id} stopped")
    
    def process_job(self, job):
        """Process a single job"""
        results = []
        
        for sequence in job.sequences:
            # Convert to tensor
            seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.system.device)
            
            # Process with optimized system
            with torch.no_grad():
                output = self.system.rl_agent.q_network(seq_tensor)
                action = torch.argmax(output).item()
                confidence = torch.softmax(output, dim=1).max().item()
            
            results.append({
                'job_id': job.job_id,
                'worker_id': self.worker_id,
                'action': action,
                'confidence': confidence,
                'timestamp': time.time()
            })
        
        return results


class CloudDeploymentPrep:
    """
    ☁️ CLOUD DEPLOYMENT PREPARATION
    Containerization and cloud-ready optimizations
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
    
    def create_docker_config(self):
        """Generate Docker configuration for deployment"""
        
        dockerfile_content = """
# Ultimate Pothole Detection - Production Deployment
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libopencv-dev \\
    python3-opencv \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/optimized_rl_model.pth
ENV OPTIMIZATION_LEVEL=speed

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "src.api.production_api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        # Save Dockerfile
        dockerfile_path = Path("Dockerfile")
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        print(f"📄 Dockerfile created: {dockerfile_path}")
        
        # Create docker-compose for scaling
        self.create_docker_compose()
        
        return dockerfile_path
    
    def create_docker_compose(self):
        """Create docker-compose for horizontal scaling"""
        
        compose_content = """
version: '3.8'

services:
  pothole-detection:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPTIMIZATION_LEVEL=speed
      - GPU_ACCELERATION=true
    volumes:
      - ./models:/app/models
      - ./results:/app/results
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    
  load-balancer:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - pothole-detection
    deploy:
      replicas: 1

  monitoring:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      
networks:
  pothole-network:
    driver: bridge
"""
        
        compose_path = Path("docker-compose.yml")
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        print(f"📄 Docker Compose created: {compose_path}")
        return compose_path
    
    def create_kubernetes_deployment(self):
        """Create Kubernetes deployment manifests"""
        
        deployment_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pothole-detection
  labels:
    app: pothole-detection
spec:
  replicas: 5
  selector:
    matchLabels:
      app: pothole-detection
  template:
    metadata:
      labels:
        app: pothole-detection
    spec:
      containers:
      - name: pothole-detection
        image: ultimate-pothole-detection:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPTIMIZATION_LEVEL
          value: "speed"
        - name: GPU_ACCELERATION
          value: "true"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: pothole-detection-service
spec:
  selector:
    app: pothole-detection
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
"""
        
        k8s_path = Path("kubernetes-deployment.yaml")
        with open(k8s_path, 'w') as f:
            f.write(deployment_yaml)
        
        print(f"📄 Kubernetes deployment created: {k8s_path}")
        return k8s_path


# Main Day 8 orchestrator
def run_day8_scalability_tests():
    """Run comprehensive Day 8 scalability tests"""
    
    print("🌐" * 60)
    print("DAY 8: SCALABILITY & MULTI-PROCESSING TESTS")
    print("🌐" * 60)
    
    model_path = "results/ultimate_comparison/models/best_ultimate_dqn_model.pth"
    
    # Test 1: Multi-GPU Processing
    print("\n🚀 TEST 1: MULTI-GPU PROCESSING")
    print("="*40)
    
    try:
        multi_gpu = MultiGPUProcessor(model_path)
        
        # Create test jobs
        test_jobs = []
        for i in range(20):
            job = ProcessingJob(
                job_id=f"job_{i:03d}",
                sequences=[np.random.rand(5, 224, 224, 3) for _ in range(3)],
                metadata={'batch_id': i // 5}
            )
            test_jobs.append(job)
        
        # Create batches
        job_batches = [test_jobs[i:i+5] for i in range(0, len(test_jobs), 5)]
        
        # Process in parallel
        start_time = time.time()
        results = multi_gpu.process_parallel(job_batches)
        end_time = time.time()
        
        print(f"✅ Multi-GPU processing complete!")
        print(f"   📊 Jobs processed: {len(results)}")
        print(f"   ⏱️ Total time: {end_time - start_time:.2f}s")
        print(f"   ⚡ Throughput: {len(results)/(end_time - start_time):.1f} jobs/sec")
        
    except Exception as e:
        print(f"❌ Multi-GPU test failed: {e}")
    
    # Test 2: Cloud Deployment Preparation
    print("\n☁️ TEST 2: CLOUD DEPLOYMENT PREPARATION")
    print("="*40)
    
    try:
        cloud_prep = CloudDeploymentPrep(model_path)
        
        # Create deployment configurations
        dockerfile = cloud_prep.create_docker_config()
        k8s_manifest = cloud_prep.create_kubernetes_deployment()
        
        print(f"✅ Cloud deployment preparation complete!")
        print(f"   📄 Docker configuration: {dockerfile}")
        print(f"   ☸️ Kubernetes manifest: {k8s_manifest}")
        
    except Exception as e:
        print(f"❌ Cloud deployment prep failed: {e}")
    
    print(f"\n🎉 DAY 8 SCALABILITY TESTS COMPLETED!")


if __name__ == "__main__":
    print("🌐 STARTING DAY 8 SCALABILITY & MULTI-PROCESSING!")
    print("="*60)
    
    # Run scalability tests
    run_day8_scalability_tests()
    
    print("🎉 DAY 8 SCALABILITY COMPLETED!")
    print("🚀 System ready for distributed deployment!")
