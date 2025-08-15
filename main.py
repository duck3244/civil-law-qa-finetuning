# main.py
# 메인 실행 스크립트

import argparse
import os
import sys
import torch
from datetime import datetime

# 모듈 임포트
from core.data_loader import CivilLawDataLoader
from core.data_formatter import DataFormatter
from core.model_setup import ModelManager
from core.trainer_config import TrainerManager
from core.inference_engine import InferenceEngine

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="민법 QA 데이터셋 Fine-tuning")
    
    # 기본 설정
    parser.add_argument("--data_file", type=str, default="civil_law_qa_dataset.csv",
                       help="데이터셋 CSV 파일 경로")
    parser.add_argument("--output_dir", type=str, default="./llama-1b-civil-law-specialist",
                       help="모델 출력 디렉토리")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                       help="베이스 모델 이름")
    
    # 모드 선택
    parser.add_argument("--mode", type=str, choices=["train", "inference", "test", "interactive"],
                       default="train", help="실행 모드")
    
    # 훈련 설정
    parser.add_argument("--preset", type=str, choices=["fast", "balanced", "quality", "memory_efficient"],
                       default="balanced", help="훈련 프리셋")
    parser.add_argument("--epochs", type=int, default=None, help="훈련 에포크 수")
    parser.add_argument("--learning_rate", type=float, default=None, help="학습률")
    parser.add_argument("--batch_size", type=int, default=None, help="배치 크기")
    
    # 데이터 필터링
    parser.add_argument("--categories", type=str, nargs="+", default=None,
                       help="사용할 카테고리 (예: 전세사기 부동산물권)")
    parser.add_argument("--difficulties", type=str, nargs="+", default=None,
                       help="사용할 난이도 (예: 초급 중급 고급)")
    
    # 추론 설정
    parser.add_argument("--model_path", type=str, default=None,
                       help="추론용 모델 경로")
    parser.add_argument("--test_file", type=str, default=None,
                       help="테스트용 질문 파일")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏛️  민법 QA 데이터셋 Fine-tuning 시스템")
    print("=" * 60)
    print(f"실행 모드: {args.mode}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if args.mode == "train":
        run_training(args)
    elif args.mode == "inference":
        run_inference(args)
    elif args.mode == "test":
        run_testing(args)
    elif args.mode == "interactive":
        run_interactive(args)

def run_training(args):
    """훈련 모드 실행"""
    print("\n🚀 훈련 모드 시작")
    
    # 1. 데이터 로딩
    print("\n📁 1단계: 데이터 로딩")
    loader = CivilLawDataLoader(args.data_file)
    df = loader.load_dataset()
    
    if df is None:
        print("❌ 데이터 로딩 실패")
        sys.exit(1)
    
    # 2. 데이터 전처리
    print("\n🔧 2단계: 데이터 전처리")
    df = loader.preprocess_data()
    
    # 카테고리/난이도 필터링
    if args.categories:
        df = loader.filter_by_category(args.categories)
    
    if args.difficulties:
        df = loader.filter_by_difficulty(args.difficulties)
    
    if len(df) == 0:
        print("❌ 필터링 후 데이터가 없습니다")
        sys.exit(1)
    
    # 3. 데이터 형식 변환
    print("\n📝 3단계: 데이터 형식 변환")
    formatter = DataFormatter()
    
    # 카테고리별 균등 분할
    train_df, eval_df = formatter.stratified_split_by_category(df)
    
    # Conversational 형식으로 변환
    train_dataset = formatter.to_conversational_format(train_df)
    eval_dataset = formatter.to_conversational_format(eval_df)
    
    print(f"훈련 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(eval_dataset)}개")
    
    # 4. 모델 설정
    print("\n🤖 4단계: 모델 설정")
    model_manager = ModelManager(args.model_name)
    model, tokenizer = model_manager.load_model_and_tokenizer()
    
    # LoRA 설정 및 적용
    peft_config = model_manager.setup_lora_config()
    model = model_manager.apply_lora()
    
    # 5. 훈련 설정
    print("\n⚙️ 5단계: 훈련 설정")
    trainer_manager = TrainerManager(args.output_dir)
    
    # 프리셋 기반 설정 생성
    custom_config = {}
    if args.epochs:
        custom_config["num_train_epochs"] = args.epochs
    if args.learning_rate:
        custom_config["learning_rate"] = args.learning_rate
    if args.batch_size:
        custom_config["per_device_train_batch_size"] = args.batch_size
    
    training_args = trainer_manager.create_training_config_from_preset(
        preset=args.preset,
        **custom_config
    )
    
    # 훈련 시간 추정
    time_estimate = trainer_manager.estimate_training_time(
        len(train_dataset), args.preset
    )
    print(f"⏱️ 예상 훈련 시간: {time_estimate['estimated_times']['default']}")
    
    # 6. 트레이너 생성 및 훈련
    print("\n🎯 6단계: 훈련 시작")
    trainer = trainer_manager.create_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer
    )
    
    # 훈련 실행
    train_result = trainer_manager.train()
    
    # 7. 모델 저장
    print("\n💾 7단계: 모델 저장")
    model_path = trainer_manager.save_model()
    
    # 훈련 정보 저장
    trainer_manager.save_training_info(
        dataset_stats=loader.get_stats(),
        model_info=model_manager.model_info
    )
    
    model_manager.save_model_config(args.output_dir)
    
    print(f"\n✅ 훈련 완료!")
    print(f"📂 모델 저장 위치: {model_path}")
    
    # 8. 간단한 테스트 (메모리 정리 개선)
    print("\n🧪 8단계: 훈련 후 테스트")
    try:
        # 철저한 메모리 정리
        print("🧹 메모리 정리 중...")
        
        # 모델과 트레이너 완전 제거
        if 'model' in locals():
            del model
        if 'trainer' in locals():
            del trainer
        if 'trainer_manager' in locals():
            del trainer_manager
        if 'model_manager' in locals():
            del model_manager
        
        # Python 가비지 컬렉션
        import gc
        gc.collect()
        
        # GPU 메모리 완전 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"🖥️ GPU 메모리 정리 완료")
            
            # 메모리 현황 출력
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 GPU 메모리: 할당 {allocated:.2f}GB, 예약 {reserved:.2f}GB")
        
        # 잠시 대기 (메모리 정리 완료 대기)
        import time
        time.sleep(2)
        
        # 새로운 프로세스에서 추론 테스트 (권장)
        print("💡 추론 테스트는 별도 프로세스에서 실행하는 것을 권장합니다:")
        print(f"   python main.py --mode test --model_path {model_path}")
        print(f"   python main.py --mode interactive --model_path {model_path}")
        
        # 간단한 CPU 테스트 시도
        print("\n🔬 CPU 모드로 간단한 테스트 시도...")
        try:
            # CPU 전용으로 테스트
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            engine = InferenceEngine()
            
            # CPU 모드로 모델 로드
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            cpu_model = PeftModel.from_pretrained(base_model, model_path)
            cpu_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            engine.model = cpu_model
            engine.tokenizer = cpu_tokenizer
            
            if engine.tokenizer.pad_token is None:
                engine.tokenizer.pad_token = engine.tokenizer.eos_token
            
            # 간단한 테스트
            test_question = "전세 계약 시 주의사항은?"
            result = engine.generate_answer(test_question, "전세사기", max_new_tokens=50)
            
            print(f"✅ CPU 테스트 성공!")
            print(f"📝 테스트 질문: {test_question}")
            print(f"🤖 모델 답변: {result['answer'][:100]}...")
            print(f"⏱️ 생성 시간: {result['generation_time']:.2f}초")
            
            # CPU 모드 정리
            del cpu_model, cpu_tokenizer, base_model, engine
            gc.collect()
            
            # CUDA 환경 복구
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            
        except Exception as cpu_error:
            print(f"⚠️ CPU 테스트도 실패: {cpu_error}")
            print("모델 파일은 정상적으로 저장되었습니다.")
            
    except Exception as e:
        print(f"⚠️ 테스트 단계 오류: {e}")
        print("✅ 훈련은 성공적으로 완료되었습니다.")
        print(f"📂 모델 저장 위치: {model_path}")
        print("\n💡 별도 프로세스에서 테스트하세요:")
        print("   1. 터미널을 새로 열고")
        print(f"   2. python main.py --mode test --model_path {model_path}")
        print("   3. 또는 python main.py --mode interactive")

def run_inference(args):
    """추론 모드 실행"""
    print("\n🔮 추론 모드 시작")
    
    model_path = args.model_path or args.output_dir
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 경로를 찾을 수 없습니다: {model_path}")
        sys.exit(1)
    
    # 추론 엔진 로드
    engine = InferenceEngine()
    engine.load_finetuned_model(model_path, args.model_name)
    
    # 대화형 추론
    print("\n💬 대화형 추론 시작")
    chat_history = engine.interactive_chat()
    
    # 채팅 기록 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chat_file = f"chat_history_{timestamp}.json"
    engine.save_test_results(chat_history, chat_file)
    print(f"채팅 기록 저장: {chat_file}")

def run_testing(args):
    """테스트 모드 실행"""
    print("\n🧪 테스트 모드 시작")
    
    model_path = args.model_path or args.output_dir
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 경로를 찾을 수 없습니다: {model_path}")
        sys.exit(1)
    
    # 추론 엔진 로드
    engine = InferenceEngine()
    engine.load_finetuned_model(model_path, args.model_name)
    
    # 종합 테스트 실행
    print("\n📊 종합 테스트 실행")
    test_results = engine.run_comprehensive_test()
    
    # 벤치마크 실행
    print("\n⚡ 성능 벤치마크 실행")
    benchmark_results = engine.benchmark_generation_speed()
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_file = f"test_results_{timestamp}.json"
    benchmark_file = f"benchmark_results_{timestamp}.json"
    
    engine.save_test_results(test_results, test_file)
    engine.save_test_results(benchmark_results, benchmark_file)
    
    print(f"✅ 테스트 완료!")
    print(f"📊 테스트 결과: {test_file}")
    print(f"⚡ 벤치마크 결과: {benchmark_file}")

def run_interactive(args):
    """대화형 모드 실행"""
    print("\n💬 대화형 모드 시작")
    
    model_path = args.model_path or args.output_dir
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 경로를 찾을 수 없습니다: {model_path}")
        print("먼저 훈련을 완료하거나 올바른 모델 경로를 지정해주세요.")
        sys.exit(1)
    
    # 추론 엔진 로드
    engine = InferenceEngine()
    engine.load_finetuned_model(model_path, args.model_name)
    
    # 대화형 채팅 시작
    chat_history = engine.interactive_chat()
    
    # 채팅 기록 저장
    if chat_history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chat_file = f"chat_history_{timestamp}.json"
        engine.save_test_results(chat_history, chat_file)
        print(f"\n채팅 기록 저장: {chat_file}")

def print_usage_examples():
    """사용 예시 출력"""
    print("\n" + "="*60)
    print("📚 사용 예시")
    print("="*60)
    print()
    print("1. 기본 훈련:")
    print("   python main.py --mode train")
    print()
    print("2. 고품질 훈련 (더 많은 에포크):")
    print("   python main.py --mode train --preset quality --epochs 6")
    print()
    print("3. 특정 카테고리만 훈련:")
    print("   python main.py --mode train --categories 전세사기")
    print()
    print("4. 추론 모드:")
    print("   python main.py --mode inference")
    print()
    print("5. 테스트 모드:")
    print("   python main.py --mode test")
    print()
    print("6. 대화형 모드:")
    print("   python main.py --mode interactive")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*60)
        print("프로그램 종료")
        print("="*60)
