# improved_main.py
# 모든 문제점이 해결된 개선된 훈련 시스템

import argparse
import os
import sys
import torch
import json
import shutil
from datetime import datetime

# 모듈 임포트
from core.data_loader import CivilLawDataLoader
from core.data_formatter import DataFormatter
from core.model_setup import ModelManager
from core.trainer_config import TrainerManager
from core.inference_engine import InferenceEngine


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="민법 QA 데이터셋 Fine-tuning (개선된 버전)")

    # 기본 설정
    parser.add_argument("--data_file", type=str, default="civil_law_qa_dataset.csv",
                        help="데이터셋 CSV 파일 경로")
    parser.add_argument("--output_dir", type=str, default="./llama-1b-civil-law-specialist-v2",
                        help="모델 출력 디렉토리")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                        help="베이스 모델 이름")

    # 모드 선택
    parser.add_argument("--mode", type=str, choices=["train", "inference", "test", "interactive"],
                        default="train", help="실행 모드")

    # 훈련 설정 (개선된 기본값)
    parser.add_argument("--preset", type=str,
                        choices=["fast", "balanced", "quality", "memory_efficient", "ultra_light"],
                        default="quality", help="훈련 프리셋")
    parser.add_argument("--epochs", type=int, default=8, help="훈련 에포크 수 (증가)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="학습률 (감소)")
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기")
    parser.add_argument("--gradient_accumulation", type=int, default=16, help="그래디언트 누적 스텝")

    # 데이터 필터링
    parser.add_argument("--categories", type=str, nargs="+", default=None,
                        help="사용할 카테고리 (예: 전세사기 부동산물권)")
    parser.add_argument("--difficulties", type=str, nargs="+", default=None,
                        help="사용할 난이도 (예: 초급 중급 고급)")

    # 추론 설정
    parser.add_argument("--model_path", type=str, default=None,
                        help="추론용 모델 경로")

    # 새로운 옵션들
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Flash Attention 사용")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="최대 시퀀스 길이")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="모델 저장 주기")
    parser.add_argument("--eval_steps", type=int, default=25,
                        help="평가 주기")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="워밍업 스텝")

    args = parser.parse_args()

    print("=" * 60)
    print("🏛️  민법 QA 데이터셋 Fine-tuning 시스템 v2.0")
    print("=" * 60)
    print(f"실행 모드: {args.mode}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if args.mode == "train":
        run_improved_training(args)
    elif args.mode == "inference":
        run_inference(args)
    elif args.mode == "test":
        run_testing(args)
    elif args.mode == "interactive":
        run_interactive(args)


def run_improved_training(args):
    """개선된 훈련 모드 실행"""
    print("\n🚀 개선된 훈련 모드 시작")

    # 출력 디렉토리 설정
    if os.path.exists(args.output_dir):
        backup_dir = f"{args.output_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(args.output_dir, backup_dir)
        print(f"📦 기존 모델 백업: {backup_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 데이터 로딩 및 전처리
    print("\n📁 1단계: 데이터 로딩 및 전처리")
    loader = CivilLawDataLoader(args.data_file)
    df = loader.load_dataset()

    if df is None:
        print("❌ 데이터 로딩 실패")
        sys.exit(1)

    # 데이터 전처리
    df = loader.preprocess_data()

    # 카테고리/난이도 필터링
    if args.categories:
        df = loader.filter_by_category(args.categories)

    if args.difficulties:
        df = loader.filter_by_difficulty(args.difficulties)

    if len(df) == 0:
        print("❌ 필터링 후 데이터가 없습니다")
        sys.exit(1)

    print(f"✅ 전처리된 데이터: {len(df)}개")

    # 2. 개선된 데이터 형식 변환
    print("\n📝 2단계: 개선된 데이터 형식 변환")
    formatter = DataFormatter()  # 이제 ImprovedDataFormatter를 가리킴

    # 데이터 증강 (품질 향상)
    df = formatter.augment_data(df, augmentation_factor=1.3)
    print(f"✅ 데이터 증강 완료: {len(df)}개")

    # 카테고리별 균등 분할 (개선된 메서드)
    train_df, eval_df = formatter.split_train_eval_stratified(df, test_size=0.2)

    # 개선된 conversational 형식으로 변환 (더 나은 프롬프트)
    train_dataset = formatter.to_conversational_format(
        train_df,
        include_category=True,
        include_difficulty=True,
        enhanced_prompts=True  # 향상된 프롬프트 사용
    )
    eval_dataset = formatter.to_conversational_format(
        eval_df,
        include_category=True,
        include_difficulty=True,
        enhanced_prompts=True
    )

    # 길이 컬럼 추가 (group_by_length용)
    train_dataset = formatter.add_length_column(train_dataset)
    eval_dataset = formatter.add_length_column(eval_dataset)

    print(f"훈련 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(eval_dataset)}개")

    # 3. 개선된 모델 설정
    print("\n🤖 3단계: 개선된 모델 설정")
    model_manager = ModelManager(args.model_name)

    # 메모리 효율적 로딩
    model, tokenizer = model_manager.load_model_and_tokenizer(
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # 메모리 절약
        attn_implementation="flash_attention_2" if args.use_flash_attention else "eager"
    )

    # 개선된 LoRA 설정
    peft_config = model_manager.setup_lora_config(
        r=32,  # 증가된 rank
        lora_alpha=64,  # 증가된 alpha
        lora_dropout=0.1,  # 증가된 dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=None,  # 문제 방지를 위해 None으로 설정
        base_model_name_or_path=args.model_name  # 명시적 지정
    )

    model = model_manager.apply_lora()

    # 4. 개선된 훈련 설정
    print("\n⚙️ 4단계: 개선된 훈련 설정")
    trainer_manager = TrainerManager(args.output_dir)

    # 안전한 훈련 설정 (지원되지 않는 파라미터 제거)
    safe_training_config = {
        # 기본 훈련 설정
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,

        # 시퀀스 및 메모리 설정
        "max_length": args.max_length,
        "gradient_checkpointing": True,
        "bf16": True,
        "dataloader_pin_memory": False,

        # 평가 및 저장 설정
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,

        # 최적화 설정
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": 1.0,

        # 로깅 설정
        "logging_steps": 10,
        "logging_first_step": True,
        "report_to": "none",

        # 안정성 설정
        "remove_unused_columns": False,
        "use_cache": False,
        "packing": False,  # 패킹 비활성화 (안정성)
        "group_by_length": False,  # length 컬럼 관련 문제 방지
    }

    # 프리셋별 추가 설정
    if args.preset == "memory_efficient":
        safe_training_config.update({
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "dataloader_pin_memory": False,
            "save_steps": 100,
            "eval_steps": 50,
        })
    elif args.preset == "quality":
        safe_training_config.update({
            "lr_scheduler_type": "cosine",
            "warmup_steps": 100,
            "save_steps": 50,
            "eval_steps": 25,
        })

    print(f"📊 훈련 설정:")
    print(f"  에포크: {safe_training_config['num_train_epochs']}")
    print(f"  학습률: {safe_training_config['learning_rate']}")
    print(f"  배치 크기: {safe_training_config['per_device_train_batch_size']}")
    print(f"  그래디언트 누적: {safe_training_config['gradient_accumulation_steps']}")
    print(f"  최대 길이: {safe_training_config['max_length']}")

    # 훈련 설정 생성
    try:
        training_args = trainer_manager.create_training_config(**safe_training_config)
        print("✅ 훈련 설정 생성 성공")
    except Exception as config_error:
        print(f"⚠️ 고급 설정 실패, 기본 설정 사용: {config_error}")

        # 최소 기본 설정
        minimal_config = {
            "num_train_epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "max_length": args.max_length,
            "eval_strategy": "steps",
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "logging_steps": 10,
            "report_to": "none",
            "remove_unused_columns": False,
        }

        training_args = trainer_manager.create_training_config(**minimal_config)
        print("✅ 기본 설정으로 훈련 설정 생성 성공")

    # 훈련 시간 추정
    time_estimate = trainer_manager.estimate_training_time(
        len(train_dataset), args.preset
    )
    print(f"⏱️ 예상 훈련 시간: {time_estimate['estimated_times']['default']}")

    # 5. 트레이너 생성 및 훈련
    print("\n🎯 5단계: 훈련 시작")

    # 개선된 포맷팅 함수
    def improved_formatting_func(example):
        """개선된 포맷팅 함수"""
        return formatter.create_enhanced_formatting_func("legal_expert")(example)

    trainer = trainer_manager.create_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        formatting_func=improved_formatting_func
    )

    # 훈련 실행
    print("🔥 훈련 시작...")
    train_result = trainer_manager.train()

    # 6. 개선된 모델 저장
    print("\n💾 6단계: 개선된 모델 저장")

    # 모델 저장
    model_path = trainer_manager.save_model()

    # 설정 파일 수정 (문제 방지)
    fix_saved_config(args.output_dir, args.model_name)

    # 훈련 정보 저장
    trainer_manager.save_training_info(
        dataset_stats=loader.get_stats(),
        model_info=model_manager.model_info
    )

    model_manager.save_model_config(args.output_dir)

    print(f"\n✅ 훈련 완료!")
    print(f"📂 모델 저장 위치: {model_path}")

    # 7. 즉시 테스트
    print("\n🧪 7단계: 즉시 테스트")

    # 메모리 정리
    cleanup_memory()

    # 간단한 테스트
    try:
        print("🔬 빠른 품질 테스트 실행...")
        quick_quality_test(args.output_dir, args.model_name)

    except Exception as e:
        print(f"⚠️ 즉시 테스트 실패: {e}")
        print("💡 별도 프로세스에서 테스트하세요:")
        print(f"   python improved_main.py --mode test --model_path {args.output_dir}")


def fix_saved_config(output_dir, base_model_name):
    """저장된 설정 파일 수정 (문제 방지)"""

    print("🔧 설정 파일 수정 중...")

    config_path = os.path.join(output_dir, "adapter_config.json")

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # 필수 수정사항
            config['base_model_name_or_path'] = base_model_name

            # modules_to_save 제거 (문제 방지)
            if 'modules_to_save' in config:
                config['modules_to_save'] = None

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            print("✅ 설정 파일 수정 완료")

        except Exception as e:
            print(f"⚠️ 설정 파일 수정 실패: {e}")


def cleanup_memory():
    """메모리 정리"""

    import gc

    # Python 가비지 컬렉션
    gc.collect()

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("🧹 메모리 정리 완료")


def quick_quality_test(model_path, base_model_name):
    """빠른 품질 테스트"""

    print("🎯 빠른 품질 테스트 실행...")

    # CPU 모드로 안전하게 테스트
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # 베이스 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )

        # Fine-tuned 모델 로드
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()

        # 테스트 질문들
        test_questions = [
            "전세 계약에서 주의할 점은 무엇인가요?",
            "임대차보호법의 주요 내용을 설명해주세요.",
            "보증금을 안전하게 보호하는 방법은?"
        ]

        print("📝 품질 테스트 결과:")

        for i, question in enumerate(test_questions, 1):
            try:
                prompt = f"질문: {question}\n답변:"
                inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=400, truncation=True)

                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=False
                    )

                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = full_text.replace(prompt, "").strip()

                print(f"\n테스트 {i}:")
                print(f"Q: {question}")
                print(f"A: {answer[:200]}...")

            except Exception as e:
                print(f"테스트 {i} 실패: {e}")

        print("\n✅ 품질 테스트 완료!")

        # CUDA 환경 복구
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

    except Exception as e:
        print(f"❌ 품질 테스트 실패: {e}")


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
    print("\n" + "=" * 60)
    print("📚 개선된 시스템 사용 예시")
    print("=" * 60)
    print()
    print("1. 고품질 훈련 (권장):")
    print("   python improved_main.py --mode train --preset quality --epochs 8")
    print()
    print("2. 빠른 훈련:")
    print("   python improved_main.py --mode train --preset fast --epochs 4")
    print()
    print("3. 메모리 절약 훈련:")
    print("   python improved_main.py --mode train --preset memory_efficient")
    print()
    print("4. 추론 모드:")
    print("   python improved_main.py --mode interactive")
    print()
    print("5. 성능 테스트:")
    print("   python improved_main.py --mode test")
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
        print("\n" + "=" * 60)
        print("프로그램 종료")
        print("=" * 60)