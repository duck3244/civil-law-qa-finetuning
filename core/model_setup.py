# fixed_model_setup.py
# 완전히 수정된 모델 설정 (core/model_setup.py 교체용)

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
from peft import LoraConfig, get_peft_model, PeftModel
import json
from datetime import datetime


class ImprovedModelManager:
    """개선된 모델 및 토크나이저 관리 클래스"""

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
                                 low_cpu_mem_usage=True,
                                 attn_implementation="eager"):
        """
        개선된 모델과 토크나이저 로드
        """
        print(f"=== 개선된 모델 로딩: {self.model_name} ===")

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🖥️ GPU 메모리 정리 완료")

        # 개선된 BitsAndBytesConfig 설정
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
            print(f"🔧 양자화 설정: {'4bit' if load_in_4bit else '8bit'}")

        try:
            # 모델 로드 (개선된 설정)
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": device_map,
                "trust_remote_code": trust_remote_code,
                "use_cache": use_cache,
                "low_cpu_mem_usage": low_cpu_mem_usage,
            }

            # 양자화 설정 추가
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            # Attention 구현 설정
            if attn_implementation != "eager":
                model_kwargs["attn_implementation"] = attn_implementation
                print(f"🔧 Attention 구현: {attn_implementation}")

            # 메모리 제한 설정 (GPU 환경에서)
            if torch.cuda.is_available() and device_map == "auto":
                model_kwargs["max_memory"] = {0: "6GB", "cpu": "8GB"}

            print("📥 베이스 모델 로딩 중...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            print("✅ 베이스 모델 로딩 성공")

        except torch.cuda.OutOfMemoryError:
            print("⚠️ GPU 메모리 부족. CPU 우선 로딩으로 재시도...")
            # CPU 우선 로딩 후 부분적 GPU 이동
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="cpu",
                quantization_config=quantization_config,
                trust_remote_code=trust_remote_code,
                use_cache=use_cache,
                low_cpu_mem_usage=True,
            )
            print("✅ CPU 모드로 베이스 모델 로딩 성공")

        except Exception as e:
            print(f"⚠️ 기본 로딩 실패, 안전 모드로 재시도: {e}")
            # 안전 모드: 기본 설정으로 로딩
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True
            )
            print("✅ 안전 모드로 베이스 모델 로딩 성공")

        # 토크나이저 로드 (개선된 설정)
        print("📝 토크나이저 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
            padding_side="right",
            truncation_side="right",
            use_fast=True  # Fast tokenizer 사용
        )

        # 패딩 토큰 설정 (개선된 방법)
        if self.tokenizer.pad_token is None:
            # EOS 토큰을 패딩으로 사용하되, 별도 ID 할당
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("✅ 토크나이저 로딩 성공")

        # 채팅 포맷 설정 (기존 템플릿 확인 후 적용)
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            print("🔧 Chat template 설정 중...")
            try:
                self.model, self.tokenizer = setup_chat_format(
                    self.model,
                    self.tokenizer,
                    format="chatml",  # 명시적 포맷 지정
                    resize_to_multiple_of=8  # 효율성을 위한 크기 조정
                )
                print("✅ Chat template 설정 완료")
            except Exception as e:
                print(f"⚠️ Chat template 설정 실패, 기본 설정 사용: {e}")
        else:
            print("✅ 기존 chat template 사용")

        # 모델 정보 수집
        self._collect_model_info()

        print(f"✅ 모델 로딩 완료")
        print(f"📊 파라미터 수: {self.model_info['num_parameters']:,}")
        print(f"📚 어휘 크기: {self.model_info['vocab_size']:,}")

        # 메모리 사용량 출력
        if torch.cuda.is_available():
            memory_info = self.get_memory_usage()
            print(f"🖥️ GPU 메모리: {memory_info['allocated_gb']:.2f}GB / {memory_info.get('reserved_gb', 'N/A')}GB")

        return self.model, self.tokenizer

    def _collect_model_info(self):
        """모델 정보 수집 (개선된 버전)"""
        if self.model and self.tokenizer:
            try:
                num_params = self.model.num_parameters()
            except:
                # 양자화된 모델의 경우 다른 방법 사용
                num_params = sum(p.numel() for p in self.model.parameters())

            self.model_info = {
                "model_name": self.model_name,
                "num_parameters": num_params,
                "vocab_size": len(self.tokenizer),
                "model_type": self.model.config.model_type,
                "hidden_size": getattr(self.model.config, 'hidden_size', 'Unknown'),
                "num_layers": getattr(self.model.config, 'num_hidden_layers', 'Unknown'),
                "num_attention_heads": getattr(self.model.config, 'num_attention_heads', 'Unknown'),
                "torch_dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else 'Unknown',
                "device": str(self.model.device) if hasattr(self.model, 'device') else 'Unknown'
            }

    def setup_lora_config(self,
                          r=32,  # 증가된 기본값
                          lora_alpha=64,  # 증가된 기본값
                          lora_dropout=0.1,  # 증가된 기본값
                          target_modules=None,
                          modules_to_save=None,  # 문제 방지를 위해 None
                          task_type="CAUSAL_LM",
                          base_model_name_or_path=None):
        """
        개선된 LoRA 설정 구성
        """

        # base_model_name_or_path 명시적 설정
        if base_model_name_or_path is None:
            base_model_name_or_path = self.model_name

        # target_modules 자동 설정 (모델에 따라)
        if target_modules is None:
            if "llama" in self.model_name.lower():
                target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
            else:
                target_modules = "all-linear"

        # modules_to_save는 None으로 설정 (문제 방지)
        # embed_tokens와 lm_head 저장 시 키 매핑 문제 발생
        modules_to_save = None

        self.peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            task_type=task_type,
            base_model_name_or_path=base_model_name_or_path,  # 명시적 설정
            bias="none",  # bias 학습 안함
            use_rslora=False,  # RSLoRA 비활성화
            use_dora=False,  # DoRA 비활성화
        )

        print(f"=== 개선된 LoRA 설정 완료 ===")
        print(f"Rank (r): {r}")
        print(f"Alpha: {lora_alpha}")
        print(f"Dropout: {lora_dropout}")
        print(f"Target modules: {target_modules}")
        print(f"Base model: {base_model_name_or_path}")
        print(f"Modules to save: {modules_to_save}")

        return self.peft_config

    def apply_lora(self):
        """모델에 LoRA 적용 (개선된 버전)"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")

        if self.peft_config is None:
            print("LoRA 설정이 없습니다. 기본 설정을 사용합니다.")
            self.setup_lora_config()

        try:
            print("🔗 LoRA 어댑터 적용 중...")
            self.model = get_peft_model(self.model, self.peft_config)
            print("✅ LoRA 어댑터 적용 성공")
        except Exception as e:
            print(f"⚠️ LoRA 적용 중 오류: {e}")
            print("🔄 안전 모드로 재시도...")

            # 더 보수적인 설정으로 재시도
            safe_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],  # 최소 모듈만
                modules_to_save=None,
                task_type="CAUSAL_LM",
                base_model_name_or_path=self.model_name
            )

            self.model = get_peft_model(self.model, safe_config)
            self.peft_config = safe_config
            print("✅ 안전 모드로 LoRA 적용 성공")

        # 훈련 가능한 파라미터 확인
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"=== LoRA 적용 완료 ===")
        print(f"훈련 가능한 파라미터: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"전체 파라미터: {total_params:,}")

        return self.model

    def get_model_size_info(self):
        """모델 크기 정보 반환 (개선된 버전)"""
        if self.model is None:
            return None

        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            # 메모리 사용량 계산
            param_size = total_params * 4 / (1024 ** 3)  # float32 기준 GB
            if hasattr(self.model, 'dtype'):
                if self.model.dtype == torch.float16 or self.model.dtype == torch.bfloat16:
                    param_size = total_params * 2 / (1024 ** 3)  # half precision

            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
                "estimated_size_gb": param_size,
                "model_dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else 'Unknown'
            }
        except Exception as e:
            print(f"⚠️ 모델 크기 정보 계산 실패: {e}")
            return None

    def save_model_config(self, output_dir):
        """모델 설정 정보 저장 (개선된 안전 버전)"""

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        def safe_json_convert(obj):
            """JSON 직렬화 안전 변환"""
            if obj is None:
                return None
            elif isinstance(obj, (int, float, str, bool)):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [safe_json_convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: safe_json_convert(v) for k, v in obj.items()}
            elif hasattr(obj, '__dict__'):
                return safe_json_convert(obj.__dict__)
            else:
                return str(obj)

        try:
            # 안전한 설정 정보 수집
            config_info = {
                "model_info": safe_json_convert(self.model_info),
                "size_info": safe_json_convert(self.get_model_size_info()),
                "model_name": self.model_name,
                "saved_at": datetime.now().isoformat(),
                "version": "2.0"
            }

            # LoRA 설정 안전하게 추가
            if self.peft_config:
                try:
                    lora_config_dict = {}
                    for key, value in self.peft_config.__dict__.items():
                        if key.startswith('_'):
                            continue
                        lora_config_dict[key] = safe_json_convert(value)

                    config_info["lora_config"] = lora_config_dict
                except Exception as lora_error:
                    print(f"⚠️ LoRA 설정 저장 중 오류: {lora_error}")
                    config_info["lora_config"] = {"error": str(lora_error)}

            # 저장
            config_path = f"{output_dir}/model_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, ensure_ascii=False, indent=2)

            print(f"✅ 모델 설정 저장: {config_path}")
            return config_path

        except Exception as e:
            print(f"⚠️ 모델 설정 저장 중 오류: {e}")

            # 최소한의 정보라도 저장 시도
            try:
                minimal_config = {
                    "model_name": self.model_name,
                    "error": str(e),
                    "saved_at": datetime.now().isoformat(),
                    "version": "2.0_minimal"
                }

                config_path = f"{output_dir}/model_config_minimal.json"
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(minimal_config, f, ensure_ascii=False, indent=2)

                print(f"📝 최소 설정 저장: {config_path}")
                return config_path

            except Exception as e2:
                print(f"❌ 설정 저장 완전 실패: {e2}")
                return None

    def prepare_for_training(self):
        """훈련을 위한 모델 준비 (개선된 버전)"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("모델과 토크나이저가 로드되지 않았습니다.")

        # 그래디언트 체크포인팅 활성화
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            try:
                self.model.gradient_checkpointing_enable()
                print("✅ 그래디언트 체크포인팅 활성화")
            except Exception as e:
                print(f"⚠️ 그래디언트 체크포인팅 실패: {e}")

        # 훈련 모드 설정
        self.model.train()

        # LoRA 레이어만 훈련 모드로 설정 (안전성)
        if hasattr(self.model, 'peft_config'):
            for name, module in self.model.named_modules():
                if 'lora' in name.lower():
                    module.train()

        print("✅ 훈련 준비 완료")
        return self.model, self.tokenizer

    def get_memory_usage(self):
        """GPU 메모리 사용량 확인 (개선된 버전)"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            reserved = torch.cuda.memory_reserved() / 1024 ** 3  # GB
            max_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "max_memory_gb": round(max_memory, 2),
                "utilization_percent": round((allocated / max_memory) * 100, 1),
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0)
            }
        else:
            return {"message": "CUDA not available"}

    def load_finetuned_model(self, model_path):
        """Fine-tuned 모델 로드 (개선된 안전 버전)"""
        print(f"=== Fine-tuned 모델 로딩: {model_path} ===")

        try:
            # 설정 파일에서 베이스 모델 확인
            config_path = os.path.join(model_path, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    adapter_config = json.load(f)

                saved_base_model = adapter_config.get('base_model_name_or_path')
                if saved_base_model and saved_base_model != 'None':
                    self.model_name = saved_base_model
                    print(f"📂 설정에서 베이스 모델 확인: {self.model_name}")

            # 베이스 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # LoRA 어댑터 로드
            model = PeftModel.from_pretrained(base_model, model_path)

            # 토크나이저 로드
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
            except:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.model = model
            self.tokenizer = tokenizer

            print(f"✅ Fine-tuned 모델 로딩 완료")
            return model, tokenizer

        except Exception as e:
            print(f"❌ Fine-tuned 모델 로딩 실패: {e}")
            raise e


# ModelManager 클래스 별칭 (기존 코드 호환성)
ModelManager = ImprovedModelManager

# 사용 예시
if __name__ == "__main__":
    # 개선된 모델 매니저 초기화
    manager = ImprovedModelManager("meta-llama/Llama-3.2-1B")

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