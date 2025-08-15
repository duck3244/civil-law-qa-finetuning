# inference_engine.py
# 추론 및 테스트 엔진

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import time
from datetime import datetime
import os

class InferenceEngine:
    """민법 전문가 모델 추론 엔진"""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = self._get_default_generation_config()
        self.system_prompts = self._get_system_prompts()

    def _get_default_generation_config(self):
        """기본 생성 설정"""
        return {
            "max_new_tokens": 400,
            "do_sample": True,
            "temperature": 0.3,  # 법률 답변이므로 낮은 온도
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": None  # 토크나이저 로드 후 설정
        }

    def _get_system_prompts(self):
        """카테고리별 시스템 프롬프트"""
        return {
            "전세사기": "당신은 대한민국 민법 전문가입니다. 특히 전세사기 및 임대차 분야에 전문성을 가지고 있으며, 전세 관련 법률 문제에 대해 정확하고 실용적인 조언을 제공합니다.",
            "부동산물권": "당신은 대한민국 민법 전문가입니다. 특히 부동산물권 분야에 전문성을 가지고 있으며, 소유권, 담보권, 용익물권 등에 대해 정확하고 자세한 법률 상담을 제공합니다.",
            "default": "당신은 대한민국 민법 전문가입니다. 민법 관련 질문에 정확하고 자세하게 답변해주세요."
        }

    def load_finetuned_model(self, model_path, base_model_name="meta-llama/Llama-3.2-1B"):
        """완전 수정된 Fine-tuned 모델 로드 메서드"""
        print(f"=== 수정된 모델 로딩: {model_path} ===")

        # 강제로 베이스 모델 지정 (저장된 설정 무시)
        print(f"🔧 강제 베이스 모델: {base_model_name}")

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🖥️ GPU 메모리 정리 완료")

        try:
            # 1단계: 베이스 모델 로드
            print("📥 베이스 모델 로딩...")

            try:
                # GPU 로딩 시도
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )

                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                device_type = "GPU"

            except Exception as gpu_error:
                print(f"⚠️ GPU 로딩 실패, CPU 모드로 전환: {gpu_error}")

                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                device_type = "CPU"

            print(f"✅ 베이스 모델 로딩 성공 ({device_type})")

            # 2단계: PEFT 어댑터 로드 (여러 방법 시도)
            print("🔗 PEFT 어댑터 로딩...")

            # 방법 1: 직접 PeftModel 로드
            try:
                self.model = PeftModel.from_pretrained(
                    base_model,
                    model_path,
                    is_trainable=False
                )
                print("✅ 방법 1 성공: 직접 PeftModel 로드")

            except Exception as peft_error:
                print(f"⚠️ 방법 1 실패: {peft_error}")

                # 방법 2: 설정 파일 임시 수정 후 로드
                try:
                    print("🔄 방법 2: 설정 파일 임시 수정")

                    import json
                    import os

                    config_path = os.path.join(model_path, "adapter_config.json")

                    # 원본 설정 백업
                    with open(config_path, 'r') as f:
                        original_config = json.load(f)

                    # 임시 수정
                    fixed_config = original_config.copy()
                    fixed_config['base_model_name_or_path'] = base_model_name

                    with open(config_path, 'w') as f:
                        json.dump(fixed_config, f, indent=2)

                    # 로드 시도
                    self.model = PeftModel.from_pretrained(
                        base_model,
                        model_path,
                        is_trainable=False
                    )

                    print("✅ 방법 2 성공: 설정 파일 수정 후 로드")

                    # 원본 설정 복구 (선택사항)
                    # with open(config_path, 'w') as f:
                    #     json.dump(original_config, f, indent=2)

                except Exception as config_error:
                    print(f"⚠️ 방법 2도 실패: {config_error}")

                    # 방법 3: 수동 어댑터 가중치 로드
                    try:
                        print("🔄 방법 3: 수동 어댑터 가중치 로드")

                        from peft import LoraConfig, get_peft_model
                        import safetensors

                        # LoRA 설정 로드 및 정리
                        lora_config_dict = original_config.copy()

                        # 문제가 될 수 있는 필드 제거
                        problematic_keys = ['base_model_name_or_path', 'revision']
                        for key in problematic_keys:
                            if key in lora_config_dict:
                                del lora_config_dict[key]

                        # LoRA 설정 생성
                        lora_config = LoraConfig(**lora_config_dict)

                        # PEFT 모델 생성
                        self.model = get_peft_model(base_model, lora_config)

                        # 어댑터 가중치 수동 로드
                        adapter_weights_path = os.path.join(model_path, "adapter_model.safetensors")

                        if os.path.exists(adapter_weights_path):
                            adapter_weights = safetensors.torch.load_file(adapter_weights_path, device="cpu")

                            # 키 이름 정리 및 매핑
                            clean_weights = {}
                            for key, value in adapter_weights.items():
                                # 중복된 'base_model' 제거
                                if key.startswith('base_model.model.base_model.model.'):
                                    clean_key = key.replace('base_model.model.base_model.model.', 'base_model.')
                                elif key.startswith('base_model.model.'):
                                    clean_key = key
                                else:
                                    clean_key = f'base_model.{key}'

                                clean_weights[clean_key] = value

                            # 가중치 로드 (strict=False로 누락된 키 무시)
                            missing_keys, unexpected_keys = self.model.load_state_dict(clean_weights, strict=False)

                            if missing_keys:
                                print(f"⚠️ 누락된 키: {len(missing_keys)}개")
                            if unexpected_keys:
                                print(f"⚠️ 예상치 못한 키: {len(unexpected_keys)}개")

                            print("✅ 방법 3 성공: 수동 어댑터 가중치 로드")
                        else:
                            raise FileNotFoundError("adapter_model.safetensors 파일을 찾을 수 없습니다")

                    except Exception as manual_error:
                        print(f"❌ 모든 PEFT 로딩 방법 실패: {manual_error}")
                        raise manual_error

            # 3단계: 토크나이저 로드
            print("📝 토크나이저 로딩...")

            try:
                # 저장된 토크나이저 시도
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                print("✅ 저장된 토크나이저 로드 성공")

            except Exception as tokenizer_error:
                print(f"⚠️ 저장된 토크나이저 실패: {tokenizer_error}")

                # 베이스 모델 토크나이저 사용
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    trust_remote_code=True
                )
                print("✅ 베이스 모델 토크나이저 로드 성공")

            # 토크나이저 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id

            # 모델을 평가 모드로 설정
            self.model.eval()

            print(f"✅ 모델 로딩 완전 성공!")
            print(f"📊 모델 타입: {type(self.model).__name__}")
            print(f"🖥️ 디바이스: {next(self.model.parameters()).device}")
            print(f"📝 토크나이저 어휘 크기: {len(self.tokenizer)}")

            return self.model, self.tokenizer

        except Exception as e:
            print(f"❌ 전체 모델 로딩 실패: {e}")
            print(f"📋 오류 타입: {type(e).__name__}")

            # 구체적인 해결책 제시
            error_str = str(e)
            if "CUDA out of memory" in error_str:
                print("💡 해결책: GPU 메모리 부족 - CPU 모드로 재시도하거나 다른 GPU 사용")
            elif "None is not a local folder" in error_str:
                print("💡 해결책: adapter_config.json의 base_model_name_or_path 수정 필요")
            elif "KeyError" in error_str or "missing" in error_str.lower():
                print("💡 해결책: 키 매핑 문제 - 어댑터 가중치 재생성 필요")
            else:
                print("💡 해결책: 모델을 다시 훈련하거나 다른 베이스 모델 시도")

            raise e

    def _load_cpu_mode_fixed(self, model_path, base_model_name):
        """CPU 모드로 모델 로드 (수정 버전)"""
        print("🔄 CPU 모드로 모델 로딩 중...")

        try:
            # CPU 전용 로딩 (강제 베이스 모델 지정)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,  # 강제로 지정된 베이스 모델 사용
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            # LoRA 어댑터 로드
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.model.eval()

            # 토크나이저 로드
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    trust_remote_code=True
                )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id

            print(f"✅ CPU 모드 로딩 완료")
            print(f"⚠️ CPU 모드는 추론 속도가 느릴 수 있습니다.")

            return self.model, self.tokenizer

        except Exception as e:
            print(f"❌ CPU 모드 로딩도 실패: {e}")
            raise e

    def generate_answer(self, question, category="default", **kwargs):
        """최종 수정된 답변 생성 메서드"""

        if self.model is None or self.tokenizer is None:
            raise ValueError("모델과 토크나이저가 로드되지 않았습니다.")

        try:
            # 1. CPU 모드 강제 전환
            if self.model.device != torch.device('cpu'):
                print("🔧 안전을 위해 CPU 모드로 전환...")
                self.model = self.model.to('cpu').float()

            # 2. 매우 간단한 프롬프트 (특수 토큰 피함)
            system_content = self.system_prompts.get(category, self.system_prompts["default"])

            # 채팅 템플릿 완전 회피
            simple_prompt = f"질문: {question}\n답변:"

            # 3. 안전한 토크나이징 (베이스 토크나이저만 사용)
            try:
                inputs = self.tokenizer(
                    simple_prompt,
                    return_tensors="pt",
                    max_length=400,
                    truncation=True,
                    padding=False,  # 패딩 비활성화
                    add_special_tokens=True
                )
            except Exception as tokenize_error:
                print(f"⚠️ 토크나이징 실패, 더 단순화: {tokenize_error}")
                inputs = self.tokenizer(
                    f"{question}",
                    return_tensors="pt",
                    max_length=200,
                    truncation=True
                )

            # 4. 토큰 ID 안전성 확보
            vocab_size = len(self.tokenizer)

            # 입력 토큰 ID 검증
            input_ids = inputs['input_ids']
            max_id = input_ids.max().item()
            min_id = input_ids.min().item()

            print(f"토큰 ID 범위: {min_id} ~ {max_id}, 어휘 크기: {vocab_size}")

            # 범위 초과 토큰 수정
            if max_id >= vocab_size:
                print(f"⚠️ 토큰 ID 범위 초과 수정: {max_id} -> {vocab_size - 1}")
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                inputs['input_ids'] = input_ids

            # 음수 토큰 ID 수정
            if min_id < 0:
                print(f"⚠️ 음수 토큰 ID 수정: {min_id} -> 0")
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                inputs['input_ids'] = input_ids

            # 5. 매우 보수적인 생성 설정
            generation_config = {
                "input_ids": inputs['input_ids'],
                "max_new_tokens": min(kwargs.get("max_new_tokens", 100), 100),
                "do_sample": False,  # 그리디 디코딩만
                "num_beams": 1,
                "early_stopping": True,
                "pad_token_id": self.tokenizer.eos_token_id,  # EOS를 패딩으로 사용
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": False,
                "output_attentions": False,
                "output_hidden_states": False,
                "return_dict_in_generate": False
            }

            # attention_mask가 있으면 추가
            if 'attention_mask' in inputs:
                generation_config["attention_mask"] = inputs['attention_mask']

            # 6. 생성 실행
            start_time = time.time()

            self.model.eval()

            print("🎯 텍스트 생성 시작...")

            with torch.no_grad():
                try:
                    outputs = self.model.generate(**generation_config)
                except Exception as gen_error:
                    print(f"⚠️ 생성 오류 발생: {gen_error}")

                    # 더 간단한 설정으로 재시도
                    simple_config = {
                        "input_ids": inputs['input_ids'],
                        "max_length": inputs['input_ids'].shape[1] + 50,
                        "do_sample": False,
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "use_cache": False
                    }

                    outputs = self.model.generate(**simple_config)

            generation_time = time.time() - start_time

            # 7. 안전한 디코딩
            try:
                # 입력 길이만큼 제거
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]

                # 생성된 토큰이 있는지 확인
                if len(generated_tokens) > 0:
                    response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                else:
                    response = "답변을 생성할 수 없습니다."

            except Exception as decode_error:
                print(f"⚠️ 디코딩 오류: {decode_error}")
                try:
                    # 전체 출력 디코딩 후 입력 제거
                    full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    input_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                    response = full_response.replace(input_text, "").strip()

                    if not response:
                        response = "답변 디코딩에 실패했습니다."

                except:
                    response = "디코딩 오류가 발생했습니다."

            print(f"✅ 생성 완료: {len(response)}자")

            return {
                "answer": response.strip(),
                "generation_time": generation_time,
                "tokens_generated": len(outputs[0]) - len(inputs['input_ids'][0]),
                "device": "cpu",
                "safe_mode": True,
                "vocab_size": vocab_size
            }

        except Exception as e:
            print(f"❌ 전체 생성 프로세스 실패: {e}")
            import traceback
            traceback.print_exc()

            return {
                "answer": f"죄송합니다. 시스템 오류로 답변을 생성할 수 없습니다. 오류: {str(e)}",
                "generation_time": 0,
                "tokens_generated": 0,
                "device": "cpu",
                "error": str(e)
            }

    def batch_generate(self, questions, categories=None, **kwargs):
        """
        배치 추론

        Args:
            questions: 질문 리스트
            categories: 카테고리 리스트 (선택사항)
            **kwargs: 추가 생성 파라미터
        """
        if categories is None:
            categories = ["default"] * len(questions)

        if len(questions) != len(categories):
            raise ValueError("질문 수와 카테고리 수가 일치하지 않습니다.")

        results = []
        total_time = 0

        for i, (question, category) in enumerate(zip(questions, categories)):
            print(f"처리 중: {i+1}/{len(questions)}")

            result = self.generate_answer(question, category, **kwargs)
            results.append({
                "question": question,
                "category": category,
                **result
            })
            total_time += result["generation_time"]

        return {
            "results": results,
            "total_time": total_time,
            "average_time": total_time / len(questions)
        }

    def interactive_chat(self):
        """대화형 채팅 모드"""
        print("=== 민법 전문가 채팅 모드 ===")
        print("종료하려면 'quit', 'exit', 또는 '종료'를 입력하세요.")
        print("카테고리를 지정하려면 '카테고리:질문' 형식으로 입력하세요.")
        print("예: 전세사기:보증금을 돌려받지 못했어요")
        print("-" * 50)

        chat_history = []

        while True:
            user_input = input("\n질문: ").strip()

            if user_input.lower() in ['quit', 'exit', '종료']:
                print("채팅을 종료합니다.")
                break

            if not user_input:
                continue

            # 카테고리 파싱
            if ':' in user_input:
                category, question = user_input.split(':', 1)
                category = category.strip()
                question = question.strip()
            else:
                category = "default"
                question = user_input

            try:
                result = self.generate_answer(question, category)

                print(f"\n[{category}] 전문가 답변:")
                print(result["answer"])
                print(f"\n(생성 시간: {result['generation_time']:.2f}초)")

                # 채팅 기록 저장
                chat_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "category": category,
                    "answer": result["answer"],
                    "generation_time": result["generation_time"]
                })

            except Exception as e:
                print(f"오류 발생: {e}")

        return chat_history

    def evaluate_on_test_cases(self, test_cases):
        """
        테스트 케이스에 대한 평가

        Args:
            test_cases: [{"question": str, "category": str, "expected": str (선택)}, ...]
        """
        print("=== 테스트 케이스 평가 ===")

        results = []

        for i, test_case in enumerate(test_cases, 1):
            question = test_case["question"]
            category = test_case.get("category", "default")
            expected = test_case.get("expected", None)

            print(f"\n{i}. [{category}] 테스트")
            print(f"질문: {question}")

            result = self.generate_answer(question, category)

            print(f"생성된 답변: {result['answer']}")

            evaluation = {
                "test_id": i,
                "question": question,
                "category": category,
                "generated_answer": result["answer"],
                "generation_time": result["generation_time"],
                "tokens_generated": result["tokens_generated"]
            }

            if expected:
                evaluation["expected_answer"] = expected
                # 간단한 유사성 평가 (실제로는 더 정교한 평가 메트릭 사용)
                similarity = self._calculate_similarity(result["answer"], expected)
                evaluation["similarity_score"] = similarity

            results.append(evaluation)
            print("-" * 80)

        return results

    def _calculate_similarity(self, generated, expected):
        """간단한 텍스트 유사성 계산 (BLEU 스코어 등을 사용할 수 있음)"""
        # 간단한 단어 기반 유사성
        gen_words = set(generated.lower().split())
        exp_words = set(expected.lower().split())

        if not exp_words:
            return 0.0

        intersection = gen_words.intersection(exp_words)
        return len(intersection) / len(exp_words)

    def get_default_test_cases(self):
        """기본 테스트 케이스 반환"""
        return [
            {
                "category": "전세사기",
                "question": "전세보증금을 돌려받지 못할 것 같은데, 어떤 법적 절차를 밟을 수 있나요?",
            },
            {
                "category": "부동산물권",
                "question": "근저당권이 설정된 부동산을 매수할 때 주의사항은 무엇인가요?",
            },
            {
                "category": "전세사기",
                "question": "전세 계약 전에 반드시 확인해야 할 서류들은 무엇인가요?",
            },
            {
                "category": "부동산물권",
                "question": "부동산 취득시효가 완성되려면 어떤 조건들이 필요한가요?",
            },
            {
                "category": "전세사기",
                "question": "깡통전세를 피하려면 어떻게 해야 하나요?",
            },
            {
                "category": "부동산물권",
                "question": "전세권과 임차권의 차이점을 설명해주세요.",
            }
        ]

    def run_comprehensive_test(self):
        """종합 테스트 실행"""
        print("=== 민법 전문가 모델 종합 테스트 ===")

        test_cases = self.get_default_test_cases()
        results = self.evaluate_on_test_cases(test_cases)

        # 통계 계산
        total_time = sum(r["generation_time"] for r in results)
        avg_time = total_time / len(results)
        total_tokens = sum(r["tokens_generated"] for r in results)
        avg_tokens = total_tokens / len(results)

        # 카테고리별 통계
        category_stats = {}
        for result in results:
            category = result["category"]
            if category not in category_stats:
                category_stats[category] = {"count": 0, "total_time": 0, "total_tokens": 0}

            category_stats[category]["count"] += 1
            category_stats[category]["total_time"] += result["generation_time"]
            category_stats[category]["total_tokens"] += result["tokens_generated"]

        # 결과 요약
        summary = {
            "test_results": results,
            "overall_stats": {
                "total_tests": len(results),
                "total_time": total_time,
                "average_time": avg_time,
                "total_tokens": total_tokens,
                "average_tokens": avg_tokens,
                "tokens_per_second": total_tokens / total_time if total_time > 0 else 0
            },
            "category_stats": {
                cat: {
                    "count": stats["count"],
                    "avg_time": stats["total_time"] / stats["count"],
                    "avg_tokens": stats["total_tokens"] / stats["count"]
                }
                for cat, stats in category_stats.items()
            },
            "timestamp": datetime.now().isoformat()
        }

        print(f"\n=== 테스트 요약 ===")
        print(f"총 테스트: {summary['overall_stats']['total_tests']}개")
        print(f"평균 생성 시간: {summary['overall_stats']['average_time']:.2f}초")
        print(f"평균 토큰 수: {summary['overall_stats']['average_tokens']:.1f}개")
        print(f"생성 속도: {summary['overall_stats']['tokens_per_second']:.1f} 토큰/초")

        return summary

    def save_test_results(self, results, output_path):
        """테스트 결과 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"테스트 결과 저장: {output_path}")
        return output_path

    def benchmark_generation_speed(self, num_tests=10):
        """생성 속도 벤치마크"""
        print(f"=== 생성 속도 벤치마크 ({num_tests}회 테스트) ===")

        test_question = "전세 계약에서 주의해야 할 점은 무엇인가요?"
        times = []
        token_counts = []

        for i in range(num_tests):
            print(f"테스트 {i+1}/{num_tests}")
            result = self.generate_answer(test_question, "전세사기")
            times.append(result["generation_time"])
            token_counts.append(result["tokens_generated"])

        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        avg_speed = avg_tokens / avg_time if avg_time > 0 else 0

        benchmark_result = {
            "num_tests": num_tests,
            "times": times,
            "token_counts": token_counts,
            "average_time": avg_time,
            "average_tokens": avg_tokens,
            "average_speed_tokens_per_sec": avg_speed,
            "min_time": min(times),
            "max_time": max(times)
        }

        print(f"평균 생성 시간: {avg_time:.2f}초")
        print(f"평균 토큰 수: {avg_tokens:.1f}개")
        print(f"평균 속도: {avg_speed:.1f} 토큰/초")
        print(f"최소/최대 시간: {min(times):.2f}초 / {max(times):.2f}초")

        return benchmark_result

    def update_generation_config(self, **kwargs):
        """생성 설정 업데이트"""
        self.generation_config.update(kwargs)
        print(f"생성 설정 업데이트: {kwargs}")

    def get_model_info(self):
        """모델 정보 반환"""
        if self.model is None:
            return None

        info = {
            "model_type": type(self.model).__name__,
            "device": str(self.model.device) if hasattr(self.model, 'device') else "Unknown",
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "Unknown",
        }

        if hasattr(self.model, 'peft_config'):
            info["peft_type"] = "LoRA"
            info["trainable_params"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return info

# 사용 예시
if __name__ == "__main__":
    # 추론 엔진 초기화
    engine = InferenceEngine()

    # Fine-tuned 모델 로드
    model_path = "./llama-1b-civil-law-specialist"
    engine.load_finetuned_model(model_path)

    # 종합 테스트 실행
    test_results = engine.run_comprehensive_test()

    # 테스트 결과 저장
    engine.save_test_results(test_results, "test_results.json")

    # 벤치마크 실행
    benchmark_results = engine.benchmark_generation_speed()

    # 대화형 모드 (선택사항)
    # chat_history = engine.interactive_chat()

    print("\n=== 추론 엔진 테스트 완료 ===")
    print("사용 방법:")
    print("1. engine.generate_answer('질문', '카테고리')")
    print("2. engine.interactive_chat()")
    print("3. engine.run_comprehensive_test()")