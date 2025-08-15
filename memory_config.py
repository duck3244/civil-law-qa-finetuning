# memory_optimizer.py
# RTX 4060 8GB 메모리 최적화 도구

import torch
import psutil
import os
import gc

class MemoryOptimizer:
    """RTX 4060 8GB를 위한 메모리 최적화 도구"""
    
    def __init__(self):
        self.gpu_total = 8.0  # RTX 4060 8GB
        self.gpu_safe_limit = 6.5  # 안전 여유분 1.5GB
        
    def check_memory_status(self):
        """현재 메모리 상태 확인"""
        status = {
            "cpu_percent": psutil.virtual_memory().percent,
            "cpu_available_gb": psutil.virtual_memory().available / 1024**3,
        }
        
        if torch.cuda.is_available():
            status.update({
                "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "gpu_available_gb": self.gpu_total - (torch.cuda.memory_allocated() / 1024**3),
            })
        
        return status
    
    def print_memory_status(self):
        """메모리 상태 출력"""
        status = self.check_memory_status()
        
        print("=" * 50)
        print("🖥️  메모리 상태")
        print("=" * 50)
        print(f"💾 CPU 메모리: {status['cpu_percent']:.1f}% 사용")
        print(f"💾 CPU 여유: {status['cpu_available_gb']:.2f}GB")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU 할당: {status['gpu_allocated_gb']:.2f}GB")
            print(f"🎮 GPU 예약: {status['gpu_reserved_gb']:.2f}GB")
            print(f"🎮 GPU 여유: {status['gpu_available_gb']:.2f}GB")
            
            if status['gpu_available_gb'] < 2.0:
                print("⚠️  GPU 메모리 부족!")
            elif status['gpu_available_gb'] < 3.0:
                print("⚠️  GPU 메모리 여유 부족")
            else:
                print("✅ GPU 메모리 여유 충분")
        else:
            print("❌ CUDA 사용 불가")
        
        print("=" * 50)
    
    def aggressive_cleanup(self):
        """적극적인 메모리 정리"""
        print("🧹 적극적인 메모리 정리 시작...")
        
        # Python 가비지 컬렉션
        collected = gc.collect()
        print(f"🗑️  Python 객체 {collected}개 정리")
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("🎮 GPU 캐시 정리 완료")
        
        # 환경 변수 최적화
        self.set_memory_env_vars()
        
        print("✅ 메모리 정리 완료")
    
    def set_memory_env_vars(self):
        """메모리 최적화 환경 변수 설정"""
        memory_configs = {
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false"
        }
        
        for key, value in memory_configs.items():
            os.environ[key] = value
            print(f"🔧 {key} = {value}")
    
    def can_load_model(self, model_size_gb=2.5):
        """모델 로딩 가능 여부 확인"""
        status = self.check_memory_status()
        
        if not torch.cuda.is_available():
            return False, "CUDA 사용 불가"
        
        available = status.get('gpu_available_gb', 0)
        
        if available >= model_size_gb:
            return True, f"GPU 로딩 가능 (여유: {available:.2f}GB)"
        else:
            return False, f"GPU 메모리 부족 (필요: {model_size_gb}GB, 여유: {available:.2f}GB)"
    
    def get_optimal_batch_size(self):
        """최적 배치 크기 추천"""
        status = self.check_memory_status()
        available = status.get('gpu_available_gb', 0)
        
        if available >= 4.0:
            return 2, "여유 충분"
        elif available >= 2.0:
            return 1, "적정 수준"
        else:
            return 1, "최소 설정 (CPU 권장)"
    
    def force_cpu_mode(self):
        """CPU 모드 강제 전환"""
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("🔄 CPU 모드로 강제 전환")
    
    def restore_gpu_mode(self):
        """GPU 모드 복구"""
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        print("🔄 GPU 모드 복구")

def check_rtx4060_compatibility():
    """RTX 4060 호환성 체크"""
    optimizer = MemoryOptimizer()
    
    print("🚀 RTX 4060 8GB 호환성 체크")
    optimizer.print_memory_status()
    
    # 모델 로딩 체크
    can_load, msg = optimizer.can_load_model(2.5)
    print(f"📊 모델 로딩 가능성: {msg}")
    
    # 배치 크기 추천
    batch_size, reason = optimizer.get_optimal_batch_size()
    print(f"📏 권장 배치 크기: {batch_size} ({reason})")
    
    # 최적화 권장사항
    print("\n💡 최적화 권장사항:")
    status = optimizer.check_memory_status()
    
    if status.get('gpu_available_gb', 0) < 2.0:
        print("   1. 다른 GPU 프로세스 종료")
        print("   2. 시스템 재부팅")
        print("   3. CPU 모드 사용")
    elif status.get('gpu_available_gb', 0) < 3.0:
        print("   1. ultra_light 프리셋 사용")
        print("   2. 배치 크기 1로 제한")
        print("   3. 메모리 조각화 방지 환경변수 설정")
    else:
        print("   ✅ 현재 상태로 훈련 가능")

if __name__ == "__main__":
    check_rtx4060_compatibility()
