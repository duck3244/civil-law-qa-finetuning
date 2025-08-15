# model_inspector.py
# 저장된 모델 구조 확인 및 디버깅 도구

import os
import json
import torch
import safetensors
from pathlib import Path

def inspect_saved_model(model_path):
    """저장된 모델의 구조와 내용 분석"""
    print(f"🔍 모델 검사: {model_path}")
    print("=" * 60)
    
    # 디렉토리 존재 확인
    if not os.path.exists(model_path):
        print(f"❌ 모델 경로가 존재하지 않습니다: {model_path}")
        return
    
    # 파일 목록 확인
    print("📁 저장된 파일들:")
    for file in os.listdir(model_path):
        file_path = os.path.join(model_path, file)
        size = os.path.getsize(file_path) / 1024 / 1024  # MB
        print(f"   {file} ({size:.2f} MB)")
    
    print("\n" + "=" * 60)
    
    # adapter_config.json 확인
    config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(config_path):
        print("⚙️ LoRA 설정:")
        with open(config_path, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                print(f"   {key}: {value}")
    else:
        print("❌ adapter_config.json 파일이 없습니다")
    
    print("\n" + "=" * 60)
    
    # adapter_model.safetensors 확인
    safetensors_path = os.path.join(model_path, "adapter_model.safetensors")
    if os.path.exists(safetensors_path):
        print("🔑 LoRA 어댑터 키들:")
        try:
            from safetensors import safe_open
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                print(f"   총 {len(keys)}개 키 발견:")
                for i, key in enumerate(sorted(keys)):
                    if i < 10:  # 처음 10개만 출력
                        print(f"   - {key}")
                    elif i == 10:
                        print(f"   ... 총 {len(keys)}개 (처음 10개만 표시)")
                        break
                
                # 문제가 되는 키 확인
                embed_keys = [k for k in keys if "embed" in k]
                if embed_keys:
                    print(f"\n🎯 Embedding 관련 키들:")
                    for key in embed_keys:
                        print(f"   - {key}")
                
        except Exception as e:
            print(f"❌ safetensors 파일 읽기 실패: {e}")
    else:
        print("❌ adapter_model.safetensors 파일이 없습니다")
    
    print("\n" + "=" * 60)
    
    # tokenizer 관련 파일 확인
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    print("📝 토크나이저 파일들:")
    for file in tokenizer_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")

def check_base_model_compatibility(base_model_name, model_path):
    """베이스 모델과 LoRA 어댑터 호환성 확인"""
    print(f"\n🔗 호환성 검사: {base_model_name}")
    print("=" * 60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        
        # 베이스 모델 설정 로드
        print("📥 베이스 모델 설정 로드 중...")
        base_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        
        print(f"🤖 모델 타입: {base_config.model_type}")
        print(f"📊 어휘 크기: {getattr(base_config, 'vocab_size', 'Unknown')}")
        print(f"🔢 레이어 수: {getattr(base_config, 'num_hidden_layers', 'Unknown')}")
        
        # LoRA 설정과 비교
        config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                lora_config = json.load(f)
            
            print(f"\n⚙️ LoRA 설정:")
            print(f"   타겟 모듈: {lora_config.get('target_modules', 'Unknown')}")
            print(f"   저장 모듈: {lora_config.get('modules_to_save', 'Unknown')}")
            print(f"   베이스 모델: {lora_config.get('base_model_name_or_path', 'Unknown')}")
            
            # 호환성 확인
            saved_base_model = lora_config.get('base_model_name_or_path', '')
            if base_model_name in saved_base_model or saved_base_model in base_model_name:
                print("✅ 베이스 모델 호환성: 일치")
            else:
                print(f"⚠️ 베이스 모델 불일치:")
                print(f"   현재 사용: {base_model_name}")
                print(f"   저장된 것: {saved_base_model}")
        
    except Exception as e:
        print(f"❌ 베이스 모델 확인 실패: {e}")

def suggest_fixes(model_path):
    """문제 해결 방안 제시"""
    print(f"\n💡 해결 방안")
    print("=" * 60)
    
    # adapter_config.json에서 정보 추출
    config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        original_model = config.get('base_model_name_or_path', '')
        
        print(f"1. 정확한 베이스 모델 사용:")
        print(f"   python main.py --mode test --model_path {model_path} --model_name {original_model}")
        
        print(f"\n2. 호환 모드로 모델 로드:")
        print(f"   python fix_model_loading.py --model_path {model_path}")
        
        print(f"\n3. 모델 재변환:")
        print(f"   python convert_model.py --input {model_path} --output {model_path}_fixed")
        
    else:
        print("❌ adapter_config.json이 없어 정확한 진단이 어렵습니다.")
        print("\n권장사항:")
        print("1. 모델을 다시 훈련")
        print("2. 저장 과정에서 오류가 있었을 가능성")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="모델 구조 검사 도구")
    parser.add_argument("--model_path", type=str, default="./llama-1b-civil-law-specialist",
                       help="검사할 모델 경로")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B",
                       help="베이스 모델 이름")
    
    args = parser.parse_args()
    
    # 모델 검사
    inspect_saved_model(args.model_path)
    
    # 호환성 검사
    check_base_model_compatibility(args.base_model, args.model_path)
    
    # 해결 방안 제시
    suggest_fixes(args.model_path)

if __name__ == "__main__":
    main()
