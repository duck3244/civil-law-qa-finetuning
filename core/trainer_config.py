# trainer_config.py
# 훈련 설정 및 SFTTrainer 관리

from trl import SFTConfig, SFTTrainer
import json
import os
from datetime import datetime

class TrainerManager:
    """SFTTrainer 설정 및 관리 클래스"""
    
    def __init__(self, output_dir="./llama-1b-civil-law-specialist"):
        self.output_dir = output_dir
        self.training_args = None
        self.trainer = None
        self.training_history = []
    
    def create_training_config(self,
                              # 기본 훈련 설정
                              num_train_epochs=4,
                              per_device_train_batch_size=2,
                              per_device_eval_batch_size=2,
                              gradient_accumulation_steps=8,
                              learning_rate=1e-4,
                              
                              # 평가 설정
                              eval_strategy="steps",
                              eval_steps=50,
                              
                              # 시퀀스 설정
                              max_length=1024,
                              packing=True,
                              
                              # 로깅 및 저장
                              logging_steps=20,
                              save_steps=100,
                              save_total_limit=3,
                              load_best_model_at_end=True,
                              metric_for_best_model="eval_loss",
                              greater_is_better=False,
                              
                              # 최적화
                              warmup_steps=50,
                              lr_scheduler_type="cosine",
                              optim="adamw_torch",
                              weight_decay=0.01,
                              
                              # Adam 설정
                              adam_beta1=0.9,
                              adam_beta2=0.999,
                              adam_epsilon=1e-8,
                              max_grad_norm=1.0,
                              
                              # 메모리 최적화
                              gradient_checkpointing=True,
                              fp16=False,
                              bf16=True,
                              dataloader_pin_memory=False,
                              dataloader_num_workers=0,
                              dataloader_persistent_workers=False,
                              
                              # 성능 향상
                              neftune_noise_alpha=5,
                              use_liger_kernel=True,
                              
                              # 기타
                              remove_unused_columns=False,
                              group_by_length=True,
                              report_to="none"):
        """
        SFT 훈련 설정 생성 (모든 파라미터 지원)
        """
        
        # 지원되지 않는 파라미터들을 필터링
        supported_params = {
            # 기본 훈련 설정
            'output_dir': self.output_dir,
            'num_train_epochs': num_train_epochs,
            'per_device_train_batch_size': per_device_train_batch_size,
            'per_device_eval_batch_size': per_device_eval_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'learning_rate': learning_rate,
            
            # 평가 설정
            'eval_strategy': eval_strategy,
            'eval_steps': eval_steps,
            
            # 시퀀스 설정
            'max_length': max_length,
            'packing': packing,
            
            # 로깅 및 저장
            'logging_steps': logging_steps,
            'save_steps': save_steps,
            'save_total_limit': save_total_limit,
            'load_best_model_at_end': load_best_model_at_end,
            'metric_for_best_model': metric_for_best_model,
            'greater_is_better': greater_is_better,
            
            # 최적화
            'warmup_steps': warmup_steps,
            'lr_scheduler_type': lr_scheduler_type,
            'optim': optim,
            'weight_decay': weight_decay,
            'adam_beta1': adam_beta1,
            'adam_beta2': adam_beta2,
            'adam_epsilon': adam_epsilon,
            'max_grad_norm': max_grad_norm,
            
            # 메모리 최적화
            'gradient_checkpointing': gradient_checkpointing,
            'fp16': fp16,
            'bf16': bf16,
            'dataloader_pin_memory': dataloader_pin_memory,
            
            # 기타
            'remove_unused_columns': remove_unused_columns,
            'group_by_length': group_by_length,
            'report_to': report_to,
        }
        
        # 조건부 파라미터 추가
        if neftune_noise_alpha is not None:
            supported_params['neftune_noise_alpha'] = neftune_noise_alpha
        
        # dataloader 설정 (SFTConfig에서 지원하는 경우만)
        try:
            # dataloader_num_workers가 지원되는지 확인
            supported_params['dataloader_num_workers'] = dataloader_num_workers
        except:
            print("⚠️ dataloader_num_workers 파라미터는 지원되지 않습니다.")
        
        try:
            supported_params['dataloader_persistent_workers'] = dataloader_persistent_workers
        except:
            print("⚠️ dataloader_persistent_workers 파라미터는 지원되지 않습니다.")
        
        try:
            if use_liger_kernel:
                supported_params['use_liger_kernel'] = use_liger_kernel
        except:
            print("⚠️ use_liger_kernel 파라미터는 지원되지 않습니다.")
        
        # SFTConfig 생성
        try:
            self.training_args = SFTConfig(**supported_params)
        except TypeError as e:
            print(f"⚠️ 일부 파라미터가 지원되지 않습니다: {e}")
            # 기본 파라미터만으로 재시도
            basic_params = {
                'output_dir': self.output_dir,
                'num_train_epochs': num_train_epochs,
                'per_device_train_batch_size': per_device_train_batch_size,
                'per_device_eval_batch_size': per_device_eval_batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'learning_rate': learning_rate,
                'eval_strategy': eval_strategy,
                'eval_steps': eval_steps,
                'max_length': max_length,
                'packing': packing,
                'logging_steps': logging_steps,
                'save_steps': save_steps,
                'save_total_limit': save_total_limit,
                'warmup_steps': warmup_steps,
                'lr_scheduler_type': lr_scheduler_type,
                'optim': optim,
                'weight_decay': weight_decay,
                'gradient_checkpointing': gradient_checkpointing,
                'bf16': bf16,
                'remove_unused_columns': remove_unused_columns,
                'group_by_length': group_by_length,
                'report_to': report_to,
            }
            self.training_args = SFTConfig(**basic_params)
        
        print("=== 훈련 설정 완료 ===")
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"에포크: {num_train_epochs}")
        print(f"배치 크기: {per_device_train_batch_size} (실질: {per_device_train_batch_size * gradient_accumulation_steps})")
        print(f"학습률: {learning_rate}")
        print(f"최대 길이: {max_length}")
        print(f"패킹: {packing}")
        print(f"옵티마이저: {optim}")
        print(f"그래디언트 체크포인팅: {gradient_checkpointing}")
        
        return self.training_args
    
    def create_trainer(self, model, train_dataset, eval_dataset=None, peft_config=None, tokenizer=None, formatting_func=None):
        """
        SFTTrainer 생성
        
        Args:
            model: 훈련할 모델
            train_dataset: 훈련 데이터셋
            eval_dataset: 검증 데이터셋 (선택사항)
            peft_config: PEFT 설정 (선택사항)
            tokenizer: 토크나이저
            formatting_func: 포맷팅 함수 (선택사항)
        """
        
        if self.training_args is None:
            raise ValueError("훈련 설정이 생성되지 않았습니다. create_training_config()를 먼저 호출하세요.")
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.trainer = SFTTrainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
            formatting_func=formatting_func,
        )
        
        print("=== SFTTrainer 생성 완료 ===")
        print(f"훈련 데이터: {len(train_dataset)}개")
        if eval_dataset:
            print(f"검증 데이터: {len(eval_dataset)}개")
        
        return self.trainer
    
    def train(self):
        """훈련 실행"""
        if self.trainer is None:
            raise ValueError("Trainer가 생성되지 않았습니다.")
        
        print("\n=== Fine-tuning 시작 ===")
        start_time = datetime.now()
        
        # 훈련 실행
        train_result = self.trainer.train()
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        # 훈련 결과 저장
        training_info = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "training_time_seconds": training_time.total_seconds(),
            "final_train_loss": float(train_result.training_loss),
            "total_steps": train_result.global_step,
        }
        
        self.training_history.append(training_info)
        
        print(f"✅ 훈련 완료!")
        print(f"훈련 시간: {training_time}")
        print(f"최종 손실: {train_result.training_loss:.4f}")
        print(f"총 스텝: {train_result.global_step}")
        
        return train_result
    
    def save_model(self):
        """모델 저장"""
        if self.trainer is None:
            raise ValueError("Trainer가 생성되지 않았습니다.")
        
        print("=== 모델 저장 중 ===")
        
        # 모델 저장
        self.trainer.save_model()
        
        # 토크나이저 저장 (deprecated 경고 해결)
        try:
            # 새로운 방식: processing_class 사용
            if hasattr(self.trainer, 'processing_class') and self.trainer.processing_class:
                self.trainer.processing_class.save_pretrained(self.output_dir)
            # 기존 방식: tokenizer 사용 (fallback)
            elif hasattr(self.trainer, 'tokenizer') and self.trainer.tokenizer:
                self.trainer.tokenizer.save_pretrained(self.output_dir)
            else:
                print("⚠️ 토크나이저를 찾을 수 없습니다.")
        except Exception as e:
            print(f"⚠️ 토크나이저 저장 중 오류: {e}")
        
        print(f"✅ 모델 저장 완료: {self.output_dir}")
        
        return self.output_dir
    
    def save_training_info(self, dataset_stats=None, model_info=None):
        """훈련 정보 종합 저장"""
        
        training_summary = {
            "training_config": {
                "output_dir": self.output_dir,
                "num_train_epochs": self.training_args.num_train_epochs,
                "learning_rate": self.training_args.learning_rate,
                "per_device_train_batch_size": self.training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": self.training_args.gradient_accumulation_steps,
                "max_length": self.training_args.max_length,
                "packing": self.training_args.packing,
            },
            "training_history": self.training_history,
            "dataset_stats": dataset_stats,
            "model_info": model_info,
            "specialization": "Korean Civil Law QA",
            "method": "LoRA + SFT",
            "saved_at": datetime.now().isoformat()
        }
        
        info_path = f"{self.output_dir}/training_summary.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_summary, f, ensure_ascii=False, indent=2)
        
        print(f"훈련 정보 저장: {info_path}")
        return info_path
    
    def get_training_config_preset(self, preset="balanced"):
        """
        사전 정의된 훈련 설정 프리셋
        
        Args:
            preset: 프리셋 타입 ("fast", "balanced", "quality", "memory_efficient", "ultra_light")
        """
        
        presets = {
            "fast": {
                "num_train_epochs": 2,
                "learning_rate": 2e-4,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "eval_steps": 100,
                "save_steps": 200,
                "max_length": 512,
            },
            
            "balanced": {
                "num_train_epochs": 4,
                "learning_rate": 1e-4,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "eval_steps": 50,
                "save_steps": 100,
                "max_length": 1024,
            },
            
            "quality": {
                "num_train_epochs": 6,
                "learning_rate": 5e-5,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "eval_steps": 25,
                "save_steps": 50,
                "max_length": 1024,
                "warmup_steps": 100,
            },
            
            "memory_efficient": {
                "num_train_epochs": 3,
                "learning_rate": 1e-4,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "eval_steps": 100,
                "save_steps": 200,
                "max_length": 512,
                "gradient_checkpointing": True,
                "dataloader_pin_memory": False,
                "bf16": True,
                "fp16": False,
                "optim": "adamw_8bit",  # 8bit optimizer 사용
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "max_grad_norm": 1.0,
                "warmup_steps": 10,
                "logging_steps": 50,
            },
            
            # RTX 4060 8GB 전용 초경량 설정
            "ultra_light": {
                "num_train_epochs": 2,
                "learning_rate": 2e-4,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "eval_steps": 200,
                "save_steps": 400,
                "max_length": 256,  # 매우 짧은 시퀀스
                "gradient_checkpointing": True,
                "dataloader_pin_memory": False,
                "bf16": True,
                "fp16": False,
                "optim": "adamw_8bit",
                "remove_unused_columns": True,
                "group_by_length": True,
                "neftune_noise_alpha": None,  # NEFTune 비활성화
                "eval_strategy": "no",  # 평가 비활성화
                "load_best_model_at_end": False,
                "save_total_limit": 1,
                "warmup_steps": 5,
                "logging_steps": 10,
            }
        }
        
        if preset not in presets:
            raise ValueError(f"알 수 없는 프리셋: {preset}. 사용 가능한 프리셋: {list(presets.keys())}")
        
        return presets[preset]
    
    def create_training_config_from_preset(self, preset="balanced", **kwargs):
        """프리셋을 기반으로 훈련 설정 생성"""
        preset_config = self.get_training_config_preset(preset)
        
        # 사용자 지정 설정으로 오버라이드
        preset_config.update(kwargs)
        
        return self.create_training_config(**preset_config)
    
    def estimate_training_time(self, dataset_size, preset="balanced"):
        """훈련 시간 추정"""
        preset_config = self.get_training_config_preset(preset)
        
        batch_size = preset_config["per_device_train_batch_size"]
        grad_accum = preset_config["gradient_accumulation_steps"]
        epochs = preset_config["num_train_epochs"]
        
        effective_batch_size = batch_size * grad_accum
        steps_per_epoch = dataset_size // effective_batch_size
        total_steps = steps_per_epoch * epochs
        
        # GPU 성능에 따른 대략적인 시간 추정 (초/스텝)
        time_per_step = {
            "RTX 4090": 0.5,
            "RTX 3090": 0.8,
            "V100": 1.0,
            "T4": 2.0,
            "default": 1.5
        }
        
        estimated_times = {}
        for gpu, time_per_step_val in time_per_step.items():
            total_time_seconds = total_steps * time_per_step_val
            hours = total_time_seconds // 3600
            minutes = (total_time_seconds % 3600) // 60
            estimated_times[gpu] = f"{int(hours)}시간 {int(minutes)}분"
        
        return {
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "estimated_times": estimated_times
        }

# 사용 예시
if __name__ == "__main__":
    # 트레이너 매니저 초기화
    trainer_manager = TrainerManager("./output")
    
    # 프리셋을 사용한 설정 생성
    config = trainer_manager.create_training_config_from_preset(
        preset="balanced",
        num_train_epochs=5  # 프리셋 오버라이드
    )
    
    # 훈련 시간 추정
    time_estimate = trainer_manager.estimate_training_time(
        dataset_size=100,
        preset="balanced"
    )
    print(f"훈련 시간 추정: {time_estimate}")
