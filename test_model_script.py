#!/usr/bin/env python3
# test_model.py
# 독립적인 모델 테스트 스크립트 (메모리 효율적)

import torch
import os
import gc
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def clear_memory():
    """메모리 완전 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def test_model_cpu(model_path, base_model_name="meta-llama/Llama-3.2-1B"):
    """CPU 모드로 모델 테스트"""
    print("🔄 CPU 모드로 모델 테스트 중...")
    
    # CPU 전용 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    try:
        # 베이스 모델 로드
        print("📥 베이스 모델 로딩...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # LoRA 어댑터 로드
        print("🔗 LoRA 어댑터 로딩...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        
        # 토크나이저 로드
        print("📝 토크나이저 로딩...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ 모델 로딩 완료!")
        
        # 테스트 질문들
        test_cases = [
            {
                "category": "전세사기",
                "question": "전세보증금을 안전하게 보호하는 방법은?"
            },
            {
                "category": "부동산물권",
                "question": "근저당권이란 무엇인가요?"
            },
            {
                "category": "전세사기", 
                "question": "깡통전세를 피하려면 어떻게 해야 하나요?"
            }
        ]
        
        print("\n=== 모델 테스트 시작 ===")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 테스트 {i}: [{test_case['category']}]")
            print(f"❓ 질문: {test_case['question']}")
            
            # 시스템 프롬프트 설정
            if test_case['category'] == "전세사기":
                system_content = "당신은 대한민국 민법 전문가입니다. 특히 전세사기 및 임대차 분야에 전문성을 가지고 있습니다."
            else:
                system_content = "당신은 대한민국 민법 전문가입니다. 특히 부동산물권 분야에 전문성을 가지고 있습니다."
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": test_case['question']}
            ]
            
            # 채팅 템플릿 적용
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 토크나이징
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # 답변 생성
            print("🤖 답변 생성 중...")
            import time
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            
            generation_time = time.time() - start_time
            
            # 결과 출력
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            print(f"💬 답변: {response.strip()}")
            print(f"⏱️ 생성 시간: {generation_time:.2f}초")
            print("-" * 80)
        
        print("\n✅ 모든 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ CPU 테스트 실패: {e}")
        return False
    
    finally:
        # 메모리 정리
        if 'model' in locals():
            del model
        if 'base_model' in locals():
            del base_model
        if 'tokenizer' in locals():
            del tokenizer
        clear_memory()

def test_model_gpu(model_path, base_model_name="meta-llama/Llama-3.2-1B"):
    """GPU 모드로 모델 테스트 (메모리 효율적)"""
    print("🔄 GPU 모드로 모델 테스트 중...")
    
    # CUDA 환경 복구
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    
    clear_memory()
    
    try:
        from transformers import BitsAndBytesConfig
        
        # 4bit 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        
        # 베이스 모델 로드
        print("📥 베이스 모델 로딩 (4bit)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "4GB", "cpu": "16GB"}
        )
        
        # LoRA 어댑터 로드
        print("🔗 LoRA 어댑터 로딩...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        
        # 토크나이저 로드
        print("📝 토크나이저 로딩...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ GPU 모델 로딩 완료!")
        
        # 간단한 테스트
        question = "전세 계약 시 주의해야 할 점은?"
        messages = [
            {"role": "system", "content": "당신은 대한민국 민법 전문가입니다."},
            {"role": "user", "content": question}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print(f"❓ 테스트 질문: {question}")
        print("🤖 답변 생성 중...")
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        print(f"💬 답변: {response.strip()}")
        print(f"⏱️ 생성 시간: {generation_time:.2f}초")
        print("✅ GPU 테스트 완료!")
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ GPU 메모리 부족: {e}")
        print("🔄 CPU 모드로 전환합니다...")
        return False
        
    except Exception as e:
        print(f"❌ GPU 테스트 실패: {e}")
        return False
    
    finally:
        if 'model' in locals():
            del model
        if 'base_model' in locals():
            del base_model
        if 'tokenizer' in locals():
            del tokenizer
        clear_memory()

def main():
    parser = argparse.ArgumentParser(description="모델 테스트 스크립트")
    parser.add_argument("--model_path", type=str, default="./llama-1b-civil-law-specialist",
                       help="Fine-tuned 모델 경로")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B",
                       help="베이스 모델 이름")
    parser.add_argument("--mode", type=str, choices=["cpu", "gpu", "auto"], default="auto",
                       help="실행 모드")
    
    args = parser.parse_args()
    
    print("🧪 독립적인 모델 테스트 시작")
    print(f"📂 모델 경로: {args.model_path}")
    print(f"🤖 베이스 모델: {args.base_model}")
    
    # 모델 경로 확인
    if not os.path.exists(args.model_path):
        print(f"❌ 모델 경로를 찾을 수 없습니다: {args.model_path}")
        return
    
    success = False
    
    if args.mode == "cpu":
        success = test_model_cpu(args.model_path, args.base_model)
    elif args.mode == "gpu":
        success = test_model_gpu(args.model_path, args.base_model)
    else:  # auto
        # GPU 먼저 시도, 실패하면 CPU
        if torch.cuda.is_available():
            print("🔄 GPU 모드부터 시도...")
            success = test_model_gpu(args.model_path, args.base_model)
            
        if not success:
            print("🔄 CPU 모드로 재시도...")
            success = test_model_cpu(args.model_path, args.base_model)
    
    if success:
        print("\n🎉 모델 테스트 성공!")
        print("💡 대화형 모드를 사용하려면:")
        print("   python main.py --mode interactive --model_path", args.model_path)
    else:
        print("\n❌ 모든 테스트 실패")

if __name__ == "__main__":
    main()
