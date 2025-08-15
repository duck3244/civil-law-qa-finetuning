# fixed_data_formatter.py
# 데이터 증강 오류 수정된 버전 (core/data_formatter.py 교체용)

from datasets import Dataset
import pandas as pd
import random
import numpy as np


class ImprovedDataFormatter:
    """개선된 데이터 형식 변환 클래스 (오류 수정)"""

    def __init__(self):
        self.system_prompts = {
            "전세사기": """당신은 대한민국 민법 전문가입니다. 
특히 전세사기 및 임대차 분야에 전문성을 가지고 있으며, 전세 관련 법률 문제에 대해 정확하고 실용적인 조언을 제공합니다.
관련 법령과 판례를 인용하여 구체적이고 도움이 되는 답변을 해주세요.""",

            "부동산물권": """당신은 대한민국 민법 전문가입니다. 
특히 부동산물권 분야에 전문성을 가지고 있으며, 소유권, 담보권, 용익물권 등에 대해 정확하고 자세한 법률 상담을 제공합니다.
민법 조문과 실무 사례를 바탕으로 전문적인 답변을 해주세요.""",

            "계약법": """당신은 대한민국 민법 전문가입니다.
계약의 성립, 효력, 이행, 해제 등 계약법 전반에 대해 전문적인 법률 조언을 제공합니다.
계약서 작성 요령과 분쟁 예방 방법도 함께 안내해주세요.""",

            "default": """당신은 대한민국 민법 전문가입니다. 
민법 관련 질문에 정확하고 자세하게 답변해주세요. 
관련 법조문과 실무적 조언을 포함하여 도움이 되는 답변을 제공해주세요."""
        }

    def augment_data(self, df, augmentation_factor=1.3):
        """
        안전한 데이터 증강 (오류 수정)

        Args:
            df: 원본 데이터프레임
            augmentation_factor: 증강 배수 (1.3 = 30% 증가)

        Returns:
            DataFrame: 증강된 데이터프레임
        """

        if len(df) == 0:
            print("⚠️ 빈 데이터프레임입니다. 증강을 건너뜁니다.")
            return df

        augmented_rows = []
        target_size = max(1, int(len(df) * (augmentation_factor - 1)))

        print(f"🔄 데이터 증강: {len(df)}개 → +{target_size}개 (총 {len(df) + target_size}개)")

        # 카테고리별로 증강 수행 (더 안전한 방법)
        if 'category' in df.columns:
            categories = df['category'].unique()

            for category in categories:
                category_df = df[df['category'] == category]
                category_target = max(1, int(len(category_df) * (augmentation_factor - 1)))

                print(f"  {category}: {len(category_df)}개 → +{category_target}개")

                for _ in range(category_target):
                    if len(category_df) > 0:  # 안전성 체크
                        # 카테고리 내에서 랜덤 선택
                        original_row = category_df.sample(n=1, random_state=random.randint(1, 10000)).iloc[0]

                        # 질문 변형
                        augmented_question = self._augment_question(original_row['question'])

                        # 답변 스타일 변형
                        augmented_answer = self._augment_answer(original_row['answer'])

                        # 새로운 행 생성
                        new_row = original_row.copy()
                        new_row['question'] = augmented_question
                        new_row['answer'] = augmented_answer

                        augmented_rows.append(new_row)
        else:
            # 카테고리가 없는 경우 전체에서 랜덤 선택
            for _ in range(target_size):
                if len(df) > 0:  # 안전성 체크
                    original_row = df.sample(n=1, random_state=random.randint(1, 10000)).iloc[0]

                    augmented_question = self._augment_question(original_row['question'])
                    augmented_answer = self._augment_answer(original_row['answer'])

                    new_row = original_row.copy()
                    new_row['question'] = augmented_question
                    new_row['answer'] = augmented_answer

                    augmented_rows.append(new_row)

        if len(augmented_rows) == 0:
            print("⚠️ 증강된 데이터가 없습니다. 원본 데이터만 사용합니다.")
            return df

        # 원본과 증강 데이터 결합
        augmented_df = pd.concat([df] + [pd.DataFrame([row]) for row in augmented_rows], ignore_index=True)

        # 안전한 셔플
        if len(augmented_df) > 1:
            augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"✅ 증강 완료: 총 {len(augmented_df)}개")
        return augmented_df

    def split_train_eval_stratified(self, df, test_size=0.2, seed=42):
        """
        안전한 계층적 분할 (오류 수정)

        Args:
            df: 원본 데이터프레임
            test_size: 검증 데이터 비율
            seed: 랜덤 시드

        Returns:
            tuple: (train_df, eval_df)
        """

        if len(df) == 0:
            print("❌ 빈 데이터프레임입니다.")
            return pd.DataFrame(), pd.DataFrame()

        if 'category' not in df.columns:
            return self._simple_split(df, test_size, seed)

        train_dfs = []
        eval_dfs = []

        print("📊 카테고리별 균등 분할:")

        categories = df['category'].unique()

        for category in categories:
            category_df = df[df['category'] == category]
            category_size = len(category_df)

            if category_size == 0:
                print(f"  ⚠️ {category}: 데이터가 없습니다. 건너뜁니다.")
                continue

            # 최소 1개는 검증 데이터로, 하지만 전체가 1개인 경우는 훈련 데이터로
            if category_size == 1:
                print(f"  {category}: 데이터 1개 → 훈련 1개, 검증 0개 (최소 크기)")
                train_dfs.append(category_df)
                # eval에는 빈 데이터프레임 추가하지 않음
            elif category_size == 2:
                print(f"  {category}: 데이터 2개 → 훈련 1개, 검증 1개")
                category_eval = category_df.sample(n=1, random_state=seed)
                category_train = category_df.drop(category_eval.index)
                train_dfs.append(category_train)
                eval_dfs.append(category_eval)
            else:
                # 3개 이상인 경우 정상적인 비율 적용
                n_eval = max(1, min(category_size - 1, int(category_size * test_size)))
                n_train = category_size - n_eval

                category_eval = category_df.sample(n=n_eval, random_state=seed)
                category_train = category_df.drop(category_eval.index)

                train_dfs.append(category_train)
                eval_dfs.append(category_eval)

                print(f"  {category}: 훈련 {n_train}개, 검증 {n_eval}개")

        # 결합
        if len(train_dfs) == 0:
            print("❌ 훈련 데이터가 없습니다.")
            return pd.DataFrame(), pd.DataFrame()

        train_df = pd.concat(train_dfs, ignore_index=True)

        if len(eval_dfs) == 0:
            print("⚠️ 검증 데이터가 없습니다. 훈련 데이터의 일부를 검증 데이터로 사용합니다.")
            # 훈련 데이터에서 최소한의 검증 데이터 분리
            if len(train_df) > 1:
                n_eval_from_train = max(1, int(len(train_df) * 0.1))  # 10%만 검증용으로
                eval_df = train_df.sample(n=n_eval_from_train, random_state=seed)
                train_df = train_df.drop(eval_df.index)
            else:
                eval_df = pd.DataFrame()
        else:
            eval_df = pd.concat(eval_dfs, ignore_index=True)

        # 안전한 셔플
        if len(train_df) > 1:
            train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        if len(eval_df) > 1:
            eval_df = eval_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        print(f"✅ 분할 완료: 훈련 {len(train_df)}개, 검증 {len(eval_df)}개")

        return train_df, eval_df

    def _simple_split(self, df, test_size, seed):
        """간단한 데이터 분할 (안전 버전)"""
        if len(df) <= 1:
            return df, pd.DataFrame()  # 데이터가 1개 이하면 전부 훈련용

        n_eval = max(1, int(len(df) * test_size))
        n_eval = min(n_eval, len(df) - 1)  # 최소 1개는 훈련용으로 남겨둠

        eval_df = df.sample(n=n_eval, random_state=seed)
        train_df = df.drop(eval_df.index)

        return train_df, eval_df

    def _augment_question(self, question):
        """질문 변형 (안전 버전)"""

        if not isinstance(question, str) or len(question.strip()) == 0:
            return question

        question = question.strip()

        question_variations = [
            # 문체 변형
            lambda q: q.replace("인가요?", "일까요?") if "인가요?" in q else q,
            lambda q: q.replace("무엇인가요?", "어떤 것인가요?") if "무엇인가요?" in q else q,
            lambda q: q.replace("어떻게", "어떤 방법으로") if "어떻게" in q else q,
            lambda q: q.replace("가능한가요?", "할 수 있나요?") if "가능한가요?" in q else q,

            # 맥락 추가 (확률적)
            lambda q: f"법률적으로 {q}" if not q.startswith(("법률", "법적")) and random.random() < 0.2 else q,
            lambda q: f"실제로 {q}" if not q.startswith("실제") and random.random() < 0.2 else q,
            lambda q: f"구체적으로 {q}" if not q.startswith("구체") and random.random() < 0.2 else q,
        ]

        # 랜덤하게 1-2개 변형 적용
        augmented = question
        num_variations = random.randint(1, 2)
        selected_variations = random.sample(question_variations, min(num_variations, len(question_variations)))

        for variation in selected_variations:
            try:
                augmented = variation(augmented)
            except:
                continue  # 변형 실패 시 건너뜀

        return augmented if augmented.strip() else question  # 빈 문자열 방지

    def _augment_answer(self, answer):
        """답변 변형 (안전 버전)"""

        if not isinstance(answer, str) or len(answer.strip()) == 0:
            return answer

        answer = answer.strip()

        # 간단한 스타일 변형 (내용은 보존)
        style_variations = [
            # 문장 연결 방식 변형 (확률적)
            lambda a: a.replace(". ", ".\n\n") if ". " in a and len(a) > 100 and random.random() < 0.3 else a,
            lambda a: a.replace("입니다.", "됩니다.") if "입니다." in a and random.random() < 0.2 else a,
            lambda a: a.replace("하시기 바랍니다.", "하세요.") if "하시기 바랍니다." in a and random.random() < 0.2 else a,
        ]

        augmented = answer
        for variation in style_variations:
            try:
                augmented = variation(augmented)
            except:
                continue  # 변형 실패 시 건너뜀

        return augmented if augmented.strip() else answer  # 빈 문자열 방지

    def to_conversational_format(self, df, include_category=True, include_difficulty=False, enhanced_prompts=True):
        """
        개선된 conversational 형식으로 변환
        """
        if len(df) == 0:
            print("⚠️ 빈 데이터프레임입니다. 빈 데이터셋을 반환합니다.")
            return Dataset.from_list([])

        formatted_data = []

        for _, row in df.iterrows():
            try:
                # 시스템 프롬프트 선택
                if include_category and 'category' in df.columns:
                    category = row['category']
                    system_content = self.system_prompts.get(category, self.system_prompts["default"])
                else:
                    system_content = self.system_prompts["default"]

                # 답변 구성
                answer = str(row['answer']).strip()
                question = str(row['question']).strip()

                # 빈 데이터 건너뛰기
                if not question or not answer:
                    continue

                # 향상된 답변 포맷팅
                if enhanced_prompts:
                    answer = self._enhance_answer_format(answer, row, include_category, include_difficulty)

                # 메타 정보 추가 (선택사항)
                if include_category and 'category' in df.columns:
                    meta_info = f"\n\n[분야: {row['category']}"
                    if include_difficulty and 'difficulty' in df.columns:
                        meta_info += f", 난이도: {row['difficulty']}"
                    meta_info += "]"
                    answer += meta_info

                # 향상된 질문 포맷팅
                question = self._enhance_question_format(question, row, include_category)

                # 메시지 구성
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]

                formatted_data.append({"messages": messages})

            except Exception as e:
                print(f"⚠️ 행 처리 중 오류 (건너뜀): {e}")
                continue

        if len(formatted_data) == 0:
            print("⚠️ 처리된 데이터가 없습니다. 빈 데이터셋을 반환합니다.")
            return Dataset.from_list([])

        return Dataset.from_list(formatted_data)

    def _enhance_question_format(self, question, row, include_category):
        """질문 포맷 향상 (안전 버전)"""

        try:
            enhanced_question = question.strip()

            # 존댓말 통일
            if not enhanced_question.endswith(('요?', '까?', '나요?', '세요?')):
                if enhanced_question.endswith('?'):
                    enhanced_question = enhanced_question[:-1] + '요?'
                elif not enhanced_question.endswith('?'):
                    enhanced_question += '요?'

            return enhanced_question

        except:
            return question  # 오류 시 원본 반환

    def _enhance_answer_format(self, answer, row, include_category, include_difficulty):
        """답변 포맷 향상 (안전 버전)"""

        try:
            enhanced_answer = answer.strip()

            # 답변 구조화 (긴 답변의 경우)
            if len(enhanced_answer) > 100:
                # 단락 구분 개선 (확률적)
                if random.random() < 0.3:
                    enhanced_answer = enhanced_answer.replace('. ', '.\n\n')
                    enhanced_answer = enhanced_answer.replace(':\n\n', ': ')

            return enhanced_answer

        except:
            return answer  # 오류 시 원본 반환

    def add_length_column(self, dataset):
        """시퀀스 길이 컬럼 추가 (group_by_length용)"""

        if len(dataset) == 0:
            return dataset

        def calculate_length(example):
            try:
                messages = example["messages"]
                total_length = 0

                for msg in messages:
                    total_length += len(str(msg.get("content", "")))

                return {"length": total_length}
            except:
                return {"length": 0}

        return dataset.map(calculate_length)

    def create_enhanced_formatting_func(self, template_type="legal_expert"):
        """
        향상된 formatting 함수 생성
        """

        if template_type == "legal_expert":
            def formatting_func(example):
                try:
                    messages = example["messages"]

                    # 시스템, 사용자, 어시스턴트 메시지 분리
                    system_msg = ""
                    user_msg = ""
                    assistant_msg = ""

                    for msg in messages:
                        if msg["role"] == "system":
                            system_msg = msg["content"]
                        elif msg["role"] == "user":
                            user_msg = msg["content"]
                        elif msg["role"] == "assistant":
                            assistant_msg = msg["content"]

                    # ChatML 형식으로 구성
                    formatted = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                    formatted += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                    formatted += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"

                    return formatted
                except:
                    # 오류 시 기본 형식 반환
                    return "질문: 기본 질문\n답변: 기본 답변"

        else:
            def formatting_func(example):
                try:
                    messages = example["messages"]
                    user_content = ""
                    assistant_content = ""

                    for msg in messages:
                        if msg["role"] == "user":
                            user_content = msg["content"]
                        elif msg["role"] == "assistant":
                            assistant_content = msg["content"]

                    return f"질문: {user_content}\n답변: {assistant_content}"
                except:
                    return "질문: 기본 질문\n답변: 기본 답변"

        return formatting_func


# DataFormatter 클래스 별칭 (기존 코드 호환성)
DataFormatter = ImprovedDataFormatter

# 사용 예시
if __name__ == "__main__":
    # 예시 데이터프레임
    import pandas as pd

    sample_data = {
        'question': [
            "전세 계약에서 임대인이 보증금을 돌려주지 않는다면 어떻게 해야 하나요?",
            "근저당권의 피담보채권 범위는 무엇인가요?",
        ],
        'answer': [
            "민법상 채무불이행에 해당하며, 내용증명 발송 후 소송을 진행할 수 있습니다.",
            "근저당권의 피담보채권에는 원본, 이자, 위약금, 손해배상금이 포함됩니다.",
        ],
        'category': ["전세사기", "부동산물권"],
        'difficulty': ["중급", "중급"]
    }

    df = pd.DataFrame(sample_data)

    # 개선된 포맷터 초기화
    formatter = ImprovedDataFormatter()

    # 안전한 데이터 증강
    augmented_df = formatter.augment_data(df, 1.3)
    print(f"증강 후 데이터: {len(augmented_df)}개")

    # 안전한 분할
    train_df, eval_df = formatter.split_train_eval_stratified(augmented_df, test_size=0.2)

    print(f"훈련 데이터: {len(train_df)}개")
    print(f"검증 데이터: {len(eval_df)}개")