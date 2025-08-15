# 🏛️ 민법 QA Fine-tuning 프로젝트

대한민국 민법 전문가 모델을 위한 Llama 1B Fine-tuning 프로젝트입니다.

## 📋 프로젝트 개요

이 프로젝트는 민법 QA 데이터셋을 사용하여 Llama 1B 모델을 fine-tuning하여 민법 전문 상담이 가능한 AI 모델을 구축합니다.

### 주요 특징
- **전문 분야**: 전세사기, 부동산물권 등 민법 전 분야
- **효율적 훈련**: LoRA를 활용한 parameter-efficient fine-tuning
- **메모리 최적화**: 4bit 양자화, gradient checkpointing 등
- **성능 향상**: NEFTune, Liger Kernel 등 최신 기법 적용

## 🗂️ 파일 구조

```
├── main.py                  # 메인 실행 스크립트
├── data_loader.py          # 데이터 로딩 및 전처리
├── data_formatter.py       # 데이터 형식 변환
├── model_setup.py          # 모델 및 토크나이저 설정
├── trainer_config.py       # 훈련 설정 및 SFTTrainer
├── inference_engine.py     # 추론 및 테스트 엔진
├── requirements.txt        # 필수 라이브러리
├── README.md              # 프로젝트 설명서
└── civil_law_qa_dataset.csv # 민법 QA 데이터셋
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd civil-law-qa-finetuning

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 라이브러리 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

`civil_law_qa_dataset.csv` 파일을 프로젝트 루트 디렉토리에 배치합니다.

### 3. 모델 훈련

```bash
# 기본 훈련
python main.py --mode train

# 고품질 훈련 (더 많은 에포크)
python main.py --mode train --preset quality --epochs 6

# 특정 카테고리만 훈련
python main.py --mode train --categories 전세사기 부동산물권
```

### 4. 모델 사용

```bash
# 대화형 모드
python main.py --mode interactive

# 테스트 모드
python main.py --mode test
```

## 📊 데이터셋 정보

### 구조
- **총 60개** 고품질 민법 QA 쌍
- **2개 카테고리**: 전세사기(32개), 부동산물권(28개)
- **3개 난이도**: 초급, 중급, 고급

### 형식
```csv
question,answer,category,difficulty
"질문 내용","답변 내용","카테고리","난이도"
```

## ⚙️ 훈련 설정

### 프리셋 옵션

| 프리셋 | 에포크 | 학습률 | 배치크기 | 설명 |
|--------|--------|--------|----------|------|
| `fast` | 2 | 2e-4 | 4 | 빠른 테스트용 |
| `balanced` | 4 | 1e-4 | 2 | 균형잡힌 설정 (기본) |
| `quality` | 6 | 5e-5 | 1 | 고품질 결과 |
| `memory_efficient` | 3 | 1e-4 | 1 | 메모리 절약 |

### 커스터마이징

```bash
# 에포크 수 변경
python main.py --mode train --epochs 8

# 학습률 변경
python main.py --mode train --learning_rate 5e-5

# 배치 크기 변경
python main.py --mode train --batch_size 4
```

## 🔧 주요 기능

### 1. 데이터 로딩 및 전처리 (`data_loader.py`)
- CSV 파일 자동 로딩 (다양한 인코딩 지원)
- 데이터 품질 검증 및 정제
- 카테고리별/난이도별 필터링
- 통계 정보 자동 생성

### 2. 데이터 형식 변환 (`data_formatter.py`)
- Conversational 형식 변환 (채팅 모델용)
- Instruction 형식 변환 (prompt-completion)
- 카테고리별 시스템 프롬프트 자동 생성
- 훈련/검증 데이터 균등 분할

### 3. 모델 설정 (`model_setup.py`)
- 자동 모델 로딩 (4bit/8bit 양자화 지원)
- LoRA 설정 및 적용
- 메모리 사용량 모니터링
- 모델 정보 저장

### 4. 훈련 관리 (`trainer_config.py`)
- SFTTrainer 자동 설정
- 다양한 프리셋 제공
- 훈련 시간 추정
- 훈련 이력 관리

### 5. 추론 엔진 (`inference_engine.py`)
- 대화형 채팅 모드
- 배치 추론 지원
- 성능 벤치마크
- 자동 테스트 케이스

## 🎯 사용 예시

### 훈련 예시

```python
from data_loader import CivilLawDataLoader
from data_formatter import DataFormatter
from model_setup import ModelManager
from trainer_config import TrainerManager

# 데이터 로딩
loader = CivilLawDataLoader("civil_law_qa_dataset.csv")
df = loader.load_dataset()
df = loader.preprocess_data()

# 데이터 변환
formatter = DataFormatter()
train_df, eval_df = formatter.stratified_split_by_category(df)
train_dataset = formatter.to_conversational_format(train_df)
eval_dataset = formatter.to_conversational_format(eval_df)

# 모델 설정
model_manager = ModelManager("meta-llama/Llama-3.2-1B")
model, tokenizer = model_manager.load_model_and_tokenizer()
model = model_manager.apply_lora()

# 훈련
trainer_manager = TrainerManager("./output")
training_args = trainer_manager.create_training_config_from_preset("balanced")
trainer = trainer_manager.create_trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)
trainer_manager.train()
trainer_manager.save_model()
```

### 추론 예시

```python
from inference_engine import InferenceEngine

# 모델 로드
engine = InferenceEngine()
engine.load_finetuned_model("./output")

# 질문 답변
result = engine.generate_answer(
    "전세보증금을 돌려받지 못했는데 어떻게 해야 하나요?",
    category="전세사기"
)
print(result["answer"])

# 대화형 모드
engine.interactive_chat()
```

## 📈 성능 및 최적화

### 메모리 최적화
- **4bit 양자화**: 메모리 사용량 ~75% 감소
- **Gradient Checkpointing**: 메모리 사용량 추가 감소
- **LoRA**: 훈련 가능한 파라미터 ~1% 수준

### 성능 향상 기법
- **NEFTune**: 노이즈 임베딩으로 성능 향상
- **Liger Kernel**: 20% 처리량 증가, 60% 메모리 감소
- **Flash Attention 2**: 긴 시퀀스 효율적 처리

### 예상 성능

| GPU | 메모리 사용량 | 훈련 시간 (60샘플) | 추론 속도 |
|-----|---------------|-------------------|-----------|
| RTX 4090 | ~8GB | ~30분 | ~15 토큰/초 |
| RTX 3090 | ~10GB | ~45분 | ~12 토큰/초 |
| V100 | ~12GB | ~60분 | ~10 토큰/초 |
| T4 | ~8GB | ~120분 | ~5 토큰/초 |

## 🧪 테스트 및 평가

### 자동 테스트
```bash
# 종합 테스트 실행
python main.py --mode test

# 성능 벤치마크
python -c "
from inference_engine import InferenceEngine
engine = InferenceEngine()
engine.load_finetuned_model('./output')
engine.benchmark_generation_speed()
"
```

### 테스트 카테고리
1. **전세사기 상담**: 보증금 반환, 계약 검토, 사기 예방
2. **부동산물권**: 소유권, 담보권, 취득시효 등
3. **일반 민법**: 계약법, 불법행위 등

## 🛠️ 문제 해결

### 일반적인 문제

#### 1. 메모리 부족
```bash
# 메모리 효율 프리셋 사용
python main.py --mode train --preset memory_efficient

# 배치 크기 감소
python main.py --mode train --batch_size 1
```

#### 2. 훈련 속도 느림
```bash
# 빠른 프리셋 사용
python main.py --mode train --preset fast

# 시퀀스 길이 감소
python main.py --mode train --max_length 512
```

#### 3. 모델 품질 개선
```bash
# 고품질 프리셋 사용
python main.py --mode train --preset quality --epochs 8

# 학습률 조정
python main.py --mode train --learning_rate 5e-5
```

### 디버깅

```python
# 데이터 검증
from data_loader import CivilLawDataLoader
loader = CivilLawDataLoader("civil_law_qa_dataset.csv")
df = loader.load_dataset()
print(loader.get_stats())

# 모델 정보 확인
from model_setup import ModelManager
manager = ModelManager()
model, tokenizer = manager.load_model_and_tokenizer()
print(manager.get_memory_usage())
```

## 📚 추가 자료

### 관련 기술
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [PEFT (Parameter Efficient Fine-tuning)](https://github.com/huggingface/peft)
- [Transformers](https://github.com/huggingface/transformers)
