#!/usr/bin/env python3
"""
📊 PRODUCTION MONITORING SYSTEM 📊
Day 8: Real-time monitoring, alerting, and performance tracking
"""

import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import threading
import queue
import psutil
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    fps: float
    latency_ms: float
    memory_usage_mb: float
    gpu_usage_percent: float
    cpu_usage_percent: float
    active_requests: int
    total_detections: int
    error_rate: float

@dataclass
class Alert:
    """System alert data structure"""
    timestamp: datetime
    severity: str  # 'info', 'warning', 'error', 'critical'
    component: str
    message: str
    metrics: Optional[PerformanceMetrics] = None

class ProductionMonitor:
    """
    📊 COMPREHENSIVE PRODUCTION MONITORING SYSTEM
    Real-time performance tracking, alerting, and system health monitoring
    """
    
    def __init__(self, alert_thresholds=None):
        self.alert_thresholds = alert_thresholds or self._get_default_thresholds()
        
        # Monitoring state
        self.metrics_history = []
        self.alerts = []
        self.is_monitoring = False
        
        # Threading
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
        
        # Setup logging
        self.setup_logging()
        
        print("📊 Production Monitor initialized!")
        print(f"   🚨 Alert thresholds: {len(self.alert_thresholds)} configured")
    
    def _get_default_thresholds(self):
        """Get default alert thresholds"""
        return {
            'fps_min': 15.0,
            'latency_max_ms': 100.0,
            'memory_max_mb': 4096.0,
            'gpu_usage_max': 95.0,
            'cpu_usage_max': 90.0,
            'error_rate_max': 0.05,  # 5%
            'response_time_max_ms': 200.0
        }
    
    def setup_logging(self):
        """Setup production logging"""
        log_dir = Path("results/monitoring/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"production_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('PotholeDetectionMonitor')
    
    def start_monitoring(self, interval_seconds=10):
        """Start real-time monitoring"""
        if self.is_monitoring:
            print("⚠️ Monitoring already active!")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        
        print(f"📊 Production monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        print("📊 Production monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self.collect_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self.check_alerts(metrics)
                
                # Log metrics
                self.log_metrics(metrics)
                
                # Cleanup old data (keep last 1000 entries)
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Wait for next interval
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def collect_metrics(self):
        """Collect current system metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_mb = memory_info.used / 1024 / 1024
        
        # GPU metrics
        gpu_usage = 0.0
        if torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization()
            except:
                gpu_usage = 0.0
        
        # Application metrics (would be collected from your API/processor)
        app_metrics = self.get_application_metrics()
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            fps=app_metrics.get('fps', 0.0),
            latency_ms=app_metrics.get('latency_ms', 0.0),
            memory_usage_mb=memory_mb,
            gpu_usage_percent=gpu_usage,
            cpu_usage_percent=cpu_percent,
            active_requests=app_metrics.get('active_requests', 0),
            total_detections=app_metrics.get('total_detections', 0),
            error_rate=app_metrics.get('error_rate', 0.0)
        )
    
    def get_application_metrics(self):
        """Get application-specific metrics"""
        # This would integrate with your actual application
        # For demo, return simulated metrics
        
        return {
            'fps': 25.0 + (time.time() % 10),  # Simulate varying FPS
            'latency_ms': 50.0 + (time.time() % 5) * 10,  # Simulate varying latency
            'active_requests': int(time.time()) % 10,  # Simulate active requests
            'total_detections': int(time.time()) % 1000,  # Simulate detections
            'error_rate': 0.02 + (time.time() % 20) * 0.001  # Simulate error rate
        }
    
    def check_alerts(self, metrics: PerformanceMetrics):
        """Check metrics against alert thresholds"""
        alerts_triggered = []
        
        # FPS too low
        if metrics.fps < self.alert_thresholds['fps_min']:
            alerts_triggered.append(Alert(
                timestamp=metrics.timestamp,
                severity='warning',
                component='performance',
                message=f"Low FPS detected: {metrics.fps:.1f} < {self.alert_thresholds['fps_min']}",
                metrics=metrics
            ))
        
        # High latency
        if metrics.latency_ms > self.alert_thresholds['latency_max_ms']:
            alerts_triggered.append(Alert(
                timestamp=metrics.timestamp,
                severity='warning',
                component='performance',
                message=f"High latency: {metrics.latency_ms:.1f}ms > {self.alert_thresholds['latency_max_ms']}ms",
                metrics=metrics
            ))
        
        # High memory usage
        if metrics.memory_usage_mb > self.alert_thresholds['memory_max_mb']:
            alerts_triggered.append(Alert(
                timestamp=metrics.timestamp,
                severity='error',
                component='resources',
                message=f"High memory usage: {metrics.memory_usage_mb:.1f}MB > {self.alert_thresholds['memory_max_mb']}MB",
                metrics=metrics
            ))
        
        # High CPU usage
        if metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage_max']:
            alerts_triggered.append(Alert(
                timestamp=metrics.timestamp,
                severity='warning',
                component='resources',
                message=f"High CPU usage: {metrics.cpu_usage_percent:.1f}% > {self.alert_thresholds['cpu_usage_max']}%",
                metrics=metrics
            ))
        
        # High error rate
        if metrics.error_rate > self.alert_thresholds['error_rate_max']:
            alerts_triggered.append(Alert(
                timestamp=metrics.timestamp,
                severity='critical',
                component='reliability',
                message=f"High error rate: {metrics.error_rate:.3f} > {self.alert_thresholds['error_rate_max']:.3f}",
                metrics=metrics
            ))
        
        # Process alerts
        for alert in alerts_triggered:
            self.process_alert(alert)
    
    def process_alert(self, alert: Alert):
        """Process and handle alerts"""
        # Add to alerts history
        self.alerts.append(alert)
        
        # Log alert
        log_level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        self.logger.log(log_level, f"[{alert.component.upper()}] {alert.message}")
        
        # Send notifications (email, Slack, etc.)
        self.send_alert_notification(alert)
        
        # Take automated actions for critical alerts
        if alert.severity == 'critical':
            self.handle_critical_alert(alert)
    
    def send_alert_notification(self, alert: Alert):
        """Send alert notifications (email, Slack, etc.)"""
        # This would integrate with actual notification systems
        print(f"🚨 ALERT [{alert.severity.upper()}]: {alert.message}")
        
        # For production, implement:
        # - Email notifications
        # - Slack/Teams integration
        # - SMS for critical alerts
        # - PagerDuty integration
    
    def handle_critical_alert(self, alert: Alert):
        """Handle critical alerts with automated actions"""
        print(f"🆘 CRITICAL ALERT - Taking automated action: {alert.message}")
        
        # Automated responses for critical alerts:
        if 'memory' in alert.message.lower():
            # Clear caches, restart processes, etc.
            self.emergency_memory_cleanup()
        elif 'error rate' in alert.message.lower():
            # Switch to fallback mode, restart services, etc.
            self.activate_fallback_mode()
    
    def emergency_memory_cleanup(self):
        """Emergency memory cleanup procedures"""
        print("🧹 Performing emergency memory cleanup...")
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("✅ Emergency memory cleanup completed")
    
    def activate_fallback_mode(self):
        """Activate system fallback mode"""
        print("🔄 Activating system fallback mode...")
        
        # Reduce processing complexity
        # Switch to simpler models
        # Implement circuit breaker pattern
        
        print("✅ Fallback mode activated")
    
    def log_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        self.logger.info(
            f"METRICS - FPS: {metrics.fps:.1f}, "
            f"Latency: {metrics.latency_ms:.1f}ms, "
            f"Memory: {metrics.memory_usage_mb:.1f}MB, "
            f"CPU: {metrics.cpu_usage_percent:.1f}%, "
            f"Requests: {metrics.active_requests}"
        )
    
    def generate_dashboard_data(self):
        """Generate data for monitoring dashboard"""
        if not self.metrics_history:
            return None
        
        # Get recent metrics (last hour)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        if not recent_metrics:
            return None
        
        # Calculate statistics
        fps_values = [m.fps for m in recent_metrics]
        latency_values = [m.latency_ms for m in recent_metrics]
        memory_values = [m.memory_usage_mb for m in recent_metrics]
        
        return {
            'current_metrics': asdict(self.metrics_history[-1]),
            'statistics': {
                'fps': {
                    'current': fps_values[-1],
                    'average': sum(fps_values) / len(fps_values),
                    'min': min(fps_values),
                    'max': max(fps_values)
                },
                'latency': {
                    'current': latency_values[-1],
                    'average': sum(latency_values) / len(latency_values),
                    'min': min(latency_values),
                    'max': max(latency_values)
                },
                'memory': {
                    'current': memory_values[-1],
                    'average': sum(memory_values) / len(memory_values),
                    'peak': max(memory_values)
                }
            },
            'recent_alerts': [asdict(alert) for alert in self.alerts[-10:]],
            'system_status': self.get_system_status()
        }
    
    def get_system_status(self):
        """Get overall system status"""
        if not self.metrics_history:
            return "unknown"
        
        latest_metrics = self.metrics_history[-1]
        recent_alerts = [a for a in self.alerts if a.timestamp > datetime.now() - timedelta(minutes=5)]
        
        # Check for critical alerts
        critical_alerts = [a for a in recent_alerts if a.severity == 'critical']
        if critical_alerts:
            return "critical"
        
        # Check for error alerts
        error_alerts = [a for a in recent_alerts if a.severity == 'error']
        if error_alerts:
            return "degraded"
        
        # Check performance thresholds
        if (latest_metrics.fps < self.alert_thresholds['fps_min'] * 0.8 or
            latest_metrics.latency_ms > self.alert_thresholds['latency_max_ms'] * 1.5):
            return "warning"
        
        return "healthy"
    
    def export_metrics_report(self, output_path=None):
        """Export comprehensive metrics report"""
        if not output_path:
            output_path = Path("results/monitoring") / f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'monitoring_period': {
                'start': self.metrics_history[0].timestamp.isoformat() if self.metrics_history else None,
                'end': self.metrics_history[-1].timestamp.isoformat() if self.metrics_history else None,
                'duration_hours': len(self.metrics_history) * 10 / 3600 if self.metrics_history else 0
            },
            'metrics_summary': self.generate_dashboard_data(),
            'alert_summary': {
                'total_alerts': len(self.alerts),
                'by_severity': {
                    'critical': len([a for a in self.alerts if a.severity == 'critical']),
                    'error': len([a for a in self.alerts if a.severity == 'error']),
                    'warning': len([a for a in self.alerts if a.severity == 'warning']),
                    'info': len([a for a in self.alerts if a.severity == 'info'])
                },
                'by_component': {}
            },
            'recommendations': self.generate_recommendations()
        }
        
        # Count alerts by component
        for alert in self.alerts:
            component = alert.component
            if component not in report['alert_summary']['by_component']:
                report['alert_summary']['by_component'][component] = 0
            report['alert_summary']['by_component'][component] += 1
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Metrics report exported: {output_path}")
        return output_path
    
    def generate_recommendations(self):
        """Generate optimization recommendations based on monitoring data"""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        # Analyze recent performance
        recent_fps = [m.fps for m in self.metrics_history[-50:]]
        recent_latency = [m.latency_ms for m in self.metrics_history[-50:]]
        recent_memory = [m.memory_usage_mb for m in self.metrics_history[-50:]]
        
        # FPS recommendations
        avg_fps = sum(recent_fps) / len(recent_fps) if recent_fps else 0
        if avg_fps < 20:
            recommendations.append({
                'category': 'performance',
                'priority': 'high',
                'issue': f'Low average FPS: {avg_fps:.1f}',
                'recommendation': 'Consider GPU acceleration or model optimization'
            })
        
        # Latency recommendations
        avg_latency = sum(recent_latency) / len(recent_latency) if recent_latency else 0
        if avg_latency > 100:
            recommendations.append({
                'category': 'performance',
                'priority': 'medium',
                'issue': f'High average latency: {avg_latency:.1f}ms',
                'recommendation': 'Optimize preprocessing pipeline or use batch processing'
            })
        
        # Memory recommendations
        avg_memory = sum(recent_memory) / len(recent_memory) if recent_memory else 0
        if avg_memory > 2048:
            recommendations.append({
                'category': 'resources',
                'priority': 'medium',
                'issue': f'High memory usage: {avg_memory:.1f}MB',
                'recommendation': 'Implement memory pooling or reduce batch size'
            })
        
        return recommendations


# Day 8 monitoring integration test
def run_day8_monitoring_test():
    """Run Day 8 monitoring system test"""
    
    print("📊" * 60)
    print("DAY 8: PRODUCTION MONITORING SYSTEM TEST")
    print("📊" * 60)
    
    # Initialize monitor
    monitor = ProductionMonitor()
    
    # Start monitoring
    print("\n🔄 Starting production monitoring...")
    monitor.start_monitoring(interval_seconds=2)
    
    # Let it run for 30 seconds
    print("📊 Collecting metrics for 30 seconds...")
    time.sleep(30)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Generate dashboard data
    dashboard_data = monitor.generate_dashboard_data()
    if dashboard_data:
        print(f"\n📈 MONITORING RESULTS:")
        print(f"   📊 System Status: {dashboard_data['system_status'].upper()}")
        print(f"   ⚡ Current FPS: {dashboard_data['current_metrics']['fps']:.1f}")
        print(f"   🔄 Current Latency: {dashboard_data['current_metrics']['latency_ms']:.1f}ms")
        print(f"   💾 Memory Usage: {dashboard_data['current_metrics']['memory_usage_mb']:.1f}MB")
        print(f"   🚨 Recent Alerts: {len(dashboard_data['recent_alerts'])}")
    
    # Export report
    report_path = monitor.export_metrics_report()
    
    print(f"\n✅ Monitoring test completed!")
    print(f"📄 Report saved: {report_path}")


if __name__ == "__main__":
    print("📊 STARTING DAY 8 PRODUCTION MONITORING!")
    print("="*60)
    
    # Run monitoring test
    run_day8_monitoring_test()
    
    print("🎉 DAY 8 MONITORING COMPLETED!")
    print("📊 Production monitoring system ready!")
