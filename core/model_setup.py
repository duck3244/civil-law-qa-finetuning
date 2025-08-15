# model_setup.py
# 모델 및 토크나이저 설정 담당

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
from peft import LoraConfig, get_peft_model, PeftModel
import json
from datetime import datetime

class ModelManager:
    """모델 및 토크나이저 관리 클래스"""
    
    def __init__(self, model_name="meta-llama/Llama-3.2-1B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self.model_info = {}
    
    def load_model_and_tokenizer(self, 
                                 load_in_4bit=True,
                                 load_in_8bit=False,
                                 torch_dtype=torch.bfloat16,
                                 device_map="auto",
                                 trust_remote_code=True,
                                 use_cache=False,
                                 low_cpu_mem_usage=True):
        """
        모델과 토크나이저 로드 (메모리 최적화)
        """
        print(f"=== 모델 로딩: {self.model_name} ===")
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🖥️ GPU 메모리 정리 완료")
        
        # BitsAndBytesConfig 설정 (메모리 최적화)
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_quant_type="nf4" if load_in_4bit else None,
                bnb_4bit_compute_dtype=torch_dtype if load_in_4bit else None,
                bnb_4bit_use_double_quant=True if load_in_4bit else False,
                bnb_4bit_quant_storage=torch_dtype if load_in_4bit else None,
            )
        
        try:
            # 모델 로드 (메모리 효율적)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                quantization_config=quantization_config,
                trust_remote_code=trust_remote_code,
                use_cache=use_cache,
                low_cpu_mem_usage=low_cpu_mem_usage,
                max_memory={0: "6GB", "cpu": "8GB"},  # GPU 메모리 제한
            )
        except torch.cuda.OutOfMemoryError:
            print("⚠️ GPU 메모리 부족. CPU 메모리 우선 로딩으로 재시도...")
            # CPU 우선 로딩 후 GPU로 이동
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map={"": "cpu"},  # 먼저 CPU에 로드
                quantization_config=quantization_config,
                trust_remote_code=trust_remote_code,
                use_cache=use_cache,
                low_cpu_mem_usage=True,
            )
            # 부분적으로 GPU로 이동
            self.model = self.model.to("cuda:0")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
            padding_side="right"
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 채팅 포맷 설정 (기존 템플릿 확인)
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            print("✅ 모델에 이미 chat template이 있습니다. 기존 템플릿을 사용합니다.")
        else:
            print("🔧 Chat template을 설정합니다.")
            try:
                self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)
            except ValueError as e:
                if "Chat template is already added" in str(e):
                    print("⚠️ Chat template이 이미 존재합니다. 기존 템플릿을 사용합니다.")
                    self.tokenizer.chat_template = None
                    self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)
                else:
                    raise e
        
        # 모델 정보 수집
        self._collect_model_info()
        
        print(f"✅ 모델 로딩 완료")
        print(f"📊 파라미터 수: {self.model_info['num_parameters']:,}")
        print(f"📚 어휘 크기: {self.model_info['vocab_size']:,}")
        
        # 메모리 사용량 출력
        if torch.cuda.is_available():
            memory_info = self.get_memory_usage()
            print(f"🖥️ GPU 메모리 사용량: {memory_info['allocated_gb']:.2f}GB / {memory_info.get('total_gb', 'N/A')}GB")
        
        return self.model, self.tokenizer
        
        # 모델 정보 수집
        self._collect_model_info()
        
        print(f"✅ 모델 로딩 완료")
        print(f"📊 파라미터 수: {self.model_info['num_parameters']:,}")
        print(f"📚 어휘 크기: {self.model_info['vocab_size']:,}")
        
        return self.model, self.tokenizer
    
    def _collect_model_info(self):
        """모델 정보 수집"""
        if self.model and self.tokenizer:
            self.model_info = {
                "model_name": self.model_name,
                "num_parameters": self.model.num_parameters(),
                "vocab_size": len(self.tokenizer),
                "model_type": self.model.config.model_type,
                "hidden_size": getattr(self.model.config, 'hidden_size', 'Unknown'),
                "num_layers": getattr(self.model.config, 'num_hidden_layers', 'Unknown'),
                "num_attention_heads": getattr(self.model.config, 'num_attention_heads', 'Unknown')
            }
    
    def setup_lora_config(self, 
                          r=16,
                          lora_alpha=32,
                          lora_dropout=0.05,
                          target_modules="all-linear",
                          modules_to_save=None,
                          task_type="CAUSAL_LM"):
        """
        LoRA 설정 구성
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha 파라미터
            lora_dropout: LoRA dropout 비율
            target_modules: 타겟 모듈
            modules_to_save: 저장할 모듈 (특수 토큰용)
            task_type: 태스크 타입
        """
        if modules_to_save is None:
            modules_to_save = ["lm_head", "embed_tokens"]
        
        self.peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            task_type=task_type,
        )
        
        print(f"=== LoRA 설정 완료 ===")
        print(f"Rank: {r}")
        print(f"Alpha: {lora_alpha}")
        print(f"Dropout: {lora_dropout}")
        print(f"Target modules: {target_modules}")
        
        return self.peft_config
    
    def apply_lora(self):
        """모델에 LoRA 적용"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        if self.peft_config is None:
            print("LoRA 설정이 없습니다. 기본 설정을 사용합니다.")
            self.setup_lora_config()
        
        self.model = get_peft_model(self.model, self.peft_config)
        
        # 훈련 가능한 파라미터 확인
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"=== LoRA 적용 완료 ===")
        print(f"훈련 가능한 파라미터: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"전체 파라미터: {total_params:,}")
        
        return self.model
    
    def get_model_size_info(self):
        """모델 크기 정보 반환"""
        if self.model is None:
            return None
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0
        }
    
    def save_model_config(self, output_dir):
        """모델 설정 정보 저장 (JSON 직렬화 안전)"""
        def convert_to_serializable(obj):
            """JSON 직렬화 가능한 형태로 변환"""
            if isinstance(obj, set):
                return list(obj)
            elif hasattr(obj, '__dict__'):
                return str(obj)
            elif hasattr(obj, '__class__'):
                return str(obj.__class__.__name__)
            else:
                return str(obj)
        
        def safe_config_extraction(config):
            """안전한 설정 추출"""
            if config is None:
                return None
            
            safe_config = {}
            for key, value in config.__dict__.items():
                try:
                    # JSON 직렬화 테스트
                    json.dumps(value)
                    safe_config[key] = value
                except (TypeError, ValueError):
                    # 직렬화 불가능한 경우 안전한 형태로 변환
                    safe_config[key] = convert_to_serializable(value)
            
            return safe_config
        
        try:
            config_info = {
                "model_info": self.model_info,
                "size_info": self.get_model_size_info(),
                "lora_config": safe_config_extraction(self.peft_config) if self.peft_config else None,
                "model_name": self.model_name,
                "saved_at": datetime.now().isoformat()
            }
            
            config_path = f"{output_dir}/model_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, ensure_ascii=False, indent=2)
            
            print(f"모델 설정 저장: {config_path}")
            return config_path
            
        except Exception as e:
            print(f"⚠️ 모델 설정 저장 중 오류: {e}")
            # 최소한의 정보라도 저장
            try:
                minimal_config = {
                    "model_name": self.model_name,
                    "error": str(e),
                    "saved_at": datetime.now().isoformat()
                }
                config_path = f"{output_dir}/model_config_minimal.json"
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(minimal_config, f, ensure_ascii=False, indent=2)
                print(f"최소 설정 저장: {config_path}")
                return config_path
            except Exception as e2:
                print(f"❌ 설정 저장 완전 실패: {e2}")
                return None
    
    def load_finetuned_model(self, model_path):
        """Fine-tuned 모델 로드"""
        print(f"=== Fine-tuned 모델 로딩: {model_path} ===")
        
        # 베이스 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )
        
        # LoRA 어댑터 로드
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print(f"✅ Fine-tuned 모델 로딩 완료")
        
        return model, tokenizer
    
    def prepare_for_training(self):
        """훈련을 위한 모델 준비"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("모델과 토크나이저가 로드되지 않았습니다.")
        
        # 그래디언트 체크포인팅 활성화
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # 훈련 모드 설정
        self.model.train()
        
        print("✅ 훈련 준비 완료")
        
        return self.model, self.tokenizer
    
    def get_memory_usage(self):
        """GPU 메모리 사용량 확인"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            
            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "device_count": torch.cuda.device_count()
            }
        else:
            return {"message": "CUDA not available"}

# 사용 예시
if __name__ == "__main__":
    # 모델 매니저 초기화
    manager = ModelManager("meta-llama/Llama-3.2-1B")
    
    # 모델과 토크나이저 로드
    model, tokenizer = manager.load_model_and_tokenizer()
    
    # LoRA 설정 및 적용
    peft_config = manager.setup_lora_config()
    model = manager.apply_lora()
    
    # 메모리 사용량 확인
    memory_info = manager.get_memory_usage()
    print(f"메모리 사용량: {memory_info}")
    
    # 모델 설정 저장
    manager.save_model_config("./model_output")