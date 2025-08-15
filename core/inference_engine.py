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
        """Fine-tuned 모델 로드 (키 매핑 문제 해결)"""
        print(f"=== Fine-tuned 모델 로딩: {model_path} ===")
        
        # 저장된 설정에서 원본 베이스 모델 확인
        config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                adapter_config = json.load(f)
            
            saved_base_model = adapter_config.get('base_model_name_or_path', base_model_name)
            print(f"📂 저장된 베이스 모델: {saved_base_model}")
            
            # 저장된 모델과 일치하지 않으면 경고
            if base_model_name != saved_base_model:
                print(f"⚠️ 베이스 모델 불일치 감지!")
                print(f"   요청된 모델: {base_model_name}")
                print(f"   저장된 모델: {saved_base_model}")
                print(f"🔄 저장된 모델로 변경하여 시도...")
                base_model_name = saved_base_model
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            allocated = torch.cuda.memory_allocated() / 1024**3
            available = torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated
            print(f"🖥️ GPU 메모리 상태: 사용 {allocated:.2f}GB, 여유 {available:.2f}GB")
            
            # 메모리 부족 시 CPU 모드로 전환
            if available < 2.0:
                print("⚠️ GPU 메모리 부족. CPU 모드로 전환합니다.")
                return self._load_cpu_mode(model_path, base_model_name)
        
        try:
            # GPU 메모리 제한적 로딩 시도
            print("🔄 GPU 메모리 제한적 로딩 시도...")
            
            # 안전한 양자화 설정
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
            
            # 베이스 모델 로드 (정확한 모델명 사용)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,  # 저장된 설정의 모델명 사용
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "3GB", "cpu": "16GB"},
            )
            
            # LoRA 어댑터 로드 (안전 모드)
            print("🔗 LoRA 어댑터 로딩...")
            try:
                self.model = PeftModel.from_pretrained(
                    base_model, 
                    model_path,
                    is_trainable=False  # 추론 모드로 명시
                )
            except KeyError as key_error:
                print(f"⚠️ 키 매핑 오류 감지: {key_error}")
                print("🔄 대안 로딩 방법 시도...")
                
                # 대안 1: 어댑터를 직접 로드
                from peft import LoraConfig
                
                # 저장된 LoRA 설정 로드
                lora_config = LoraConfig.from_pretrained(model_path)
                
                # 새로운 PEFT 모델 생성 후 가중치 로드
                from peft import get_peft_model
                self.model = get_peft_model(base_model, lora_config)
                
                # 어댑터 가중치 수동 로드
                import safetensors
                adapter_weights_path = os.path.join(model_path, "adapter_model.safetensors")
                if os.path.exists(adapter_weights_path):
                    adapter_weights = safetensors.torch.load_file(adapter_weights_path)
                    self.model.load_state_dict(adapter_weights, strict=False)
                    print("✅ 대안 로딩 성공")
                else:
                    raise Exception("어댑터 가중치 파일을 찾을 수 없습니다")
            
            self.model.eval()
            
        except Exception as gpu_error:
            print(f"❌ GPU 로딩 실패: {gpu_error}")
            print("🔄 CPU 모드로 fallback...")
            return self._load_cpu_mode(model_path, base_model_name)
        
        # 토크나이저 로드
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,  # 저장된 토크나이저 사용
                trust_remote_code=True
            )
        except:
            # 저장된 토크나이저가 없으면 베이스 모델 토크나이저 사용
            print("⚠️ 저장된 토크나이저가 없습니다. 베이스 모델 토크나이저 사용...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
        
        print(f"✅ GPU 모드 로딩 완료")
        print(f"📊 모델 데이터 타입: {next(self.model.parameters()).dtype}")
        print(f"🖥️ 모델 디바이스: {next(self.model.parameters()).device}")
        
        return self.model, self.tokenizer
    
    def _load_cpu_mode(self, model_path, base_model_name):
        """CPU 모드로 모델 로드"""
        print("🔄 CPU 모드로 모델 로딩 중...")
        
        try:
            # CPU 전용 로딩
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # LoRA 어댑터 로드
            self.model = PeftModel.from_pretrained(base_model, model_path)
            self.model.eval()
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
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
        """
        질문에 대한 답변 생성
        
        Args:
            question: 질문 텍스트
            category: 카테고리 ("전세사기", "부동산물권", "default")
            **kwargs: 추가 생성 파라미터
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("모델과 토크나이저가 로드되지 않았습니다.")
        
        # 시스템 프롬프트 선택
        system_content = self.system_prompts.get(category, self.system_prompts["default"])
        
        # 메시지 구성
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ]
        
        # 채팅 템플릿 적용
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 토크나이징 (디바이스와 데이터 타입 일치)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 모델과 동일한 디바이스로 이동
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 생성 설정 업데이트
        generation_config = self.generation_config.copy()
        generation_config.update(kwargs)
        
        # 모델을 evaluation 모드로 설정
        self.model.eval()
        
        # 답변 생성
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # use_cache를 False로 설정하여 gradient checkpointing 문제 해결
                outputs = self.model.generate(
                    **inputs,
                    use_cache=False,  # gradient checkpointing과 호환성을 위해
                    **generation_config
                )
        except RuntimeError as e:
            if "dtype" in str(e) or "float" in str(e):
                print("⚠️ 데이터 타입 문제 감지. 모델을 float32로 변환 후 재시도...")
                # 모델을 float32로 변환하여 재시도
                self.model = self.model.float()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        use_cache=False,
                        **generation_config
                    )
            else:
                raise e
        
        generation_time = time.time() - start_time
        
        # 디코딩 (입력 부분 제거)
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return {
            "answer": response.strip(),
            "generation_time": generation_time,
            "tokens_generated": len(outputs[0]) - len(inputs.input_ids[0])
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