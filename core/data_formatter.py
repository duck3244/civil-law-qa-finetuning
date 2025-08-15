# data_formatter.py
# 데이터 형식 변환 담당

from datasets import Dataset
import pandas as pd

class DataFormatter:
    """데이터 형식 변환 클래스"""
    
    def __init__(self):
        self.system_prompts = {
            "전세사기": "당신은 대한민국 민법 전문가입니다. 특히 전세사기 및 임대차 분야에 전문성을 가지고 있으며, 전세 관련 법률 문제에 대해 정확하고 실용적인 조언을 제공합니다.",
            "부동산물권": "당신은 대한민국 민법 전문가입니다. 특히 부동산물권 분야에 전문성을 가지고 있으며, 소유권, 담보권, 용익물권 등에 대해 정확하고 자세한 법률 상담을 제공합니다.",
            "default": "당신은 대한민국 민법 전문가입니다. 민법 관련 질문에 정확하고 자세하게 답변해주세요."
        }
    
    def to_conversational_format(self, df, include_category=True, include_difficulty=False):
        """
        데이터프레임을 conversational 형식으로 변환
        
        Args:
            df: 원본 데이터프레임
            include_category: 카테고리 정보 포함 여부
            include_difficulty: 난이도 정보 포함 여부
        
        Returns:
            Dataset: HuggingFace Dataset 객체
        """
        formatted_data = []
        
        for _, row in df.iterrows():
            # 시스템 프롬프트 선택
            if include_category and 'category' in df.columns:
                category = row['category']
                system_content = self.system_prompts.get(category, self.system_prompts["default"])
            else:
                system_content = self.system_prompts["default"]
            
            # 답변 구성
            answer = row['answer']
            
            # 메타 정보 추가 (선택사항)
            if include_category and 'category' in df.columns:
                answer += f"\n\n[분야: {row['category']}"
                if include_difficulty and 'difficulty' in df.columns:
                    answer += f", 난이도: {row['difficulty']}"
                answer += "]"
            
            # 메시지 구성
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": row['question']},
                {"role": "assistant", "content": answer}
            ]
            
            formatted_data.append({"messages": messages})
        
        return Dataset.from_list(formatted_data)
    
    def to_instruction_format(self, df, include_category=True):
        """
        데이터프레임을 instruction(prompt-completion) 형식으로 변환
        
        Args:
            df: 원본 데이터프레임
            include_category: 카테고리 정보 포함 여부
        
        Returns:
            Dataset: HuggingFace Dataset 객체
        """
        formatted_data = []
        
        for _, row in df.iterrows():
            # 프롬프트 구성
            if include_category and 'category' in df.columns:
                category = row['category']
                prompt = f"[{category} 분야 질문]\n{row['question']}"
            else:
                prompt = row['question']
            
            completion = row['answer']
            
            formatted_data.append({
                "prompt": prompt,
                "completion": completion
            })
        
        return Dataset.from_list(formatted_data)
    
    def to_custom_format(self, df, template=None):
        """
        커스텀 템플릿을 사용한 형식 변환
        
        Args:
            df: 원본 데이터프레임
            template: 커스텀 템플릿 함수
        
        Returns:
            Dataset: HuggingFace Dataset 객체
        """
        if template is None:
            template = self._default_template
        
        formatted_data = []
        
        for _, row in df.iterrows():
            formatted_text = template(row)
            formatted_data.append({"text": formatted_text})
        
        return Dataset.from_list(formatted_data)
    
    def _default_template(self, row):
        """기본 커스텀 템플릿"""
        category_info = f"[{row['category']}] " if 'category' in row else ""
        return f"### Question: {category_info}{row['question']}\n### Answer: {row['answer']}"
    
    def create_formatting_func(self, template_type="qa"):
        """
        SFTTrainer에서 사용할 formatting 함수 생성
        
        Args:
            template_type: 템플릿 타입 ("qa", "legal", "detailed")
        
        Returns:
            function: formatting 함수
        """
        if template_type == "qa":
            def formatting_func(example):
                return f"### Question: {example['question']}\n### Answer: {example['answer']}"
        
        elif template_type == "legal":
            def formatting_func(example):
                category = example.get('category', '일반')
                return f"### [{category}] 법률 상담\n질문: {example['question']}\n답변: {example['answer']}"
        
        elif template_type == "detailed":
            def formatting_func(example):
                category = example.get('category', '일반')
                difficulty = example.get('difficulty', '중급')
                return f"### 민법 상담 [{category} | {difficulty}]\n질문: {example['question']}\n전문가 답변: {example['answer']}"
        
        else:
            def formatting_func(example):
                return f"{example['question']}\n{example['answer']}"
        
        return formatting_func
    
    def split_train_eval(self, dataset, test_size=0.15, seed=42):
        """
        훈련/검증 데이터 분할
        
        Args:
            dataset: Dataset 객체
            test_size: 검증 데이터 비율
            seed: 랜덤 시드
        
        Returns:
            tuple: (train_dataset, eval_dataset)
        """
        split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
        return split_dataset['train'], split_dataset['test']
    
    def stratified_split_by_category(self, df, test_size=0.15, seed=42):
        """
        카테고리별 균등 분할
        
        Args:
            df: 원본 데이터프레임
            test_size: 검증 데이터 비율
            seed: 랜덤 시드
        
        Returns:
            tuple: (train_df, eval_df)
        """
        if 'category' not in df.columns:
            # 카테고리가 없으면 일반 분할
            return self._simple_split(df, test_size, seed)
        
        train_dfs = []
        eval_dfs = []
        
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            
            # 카테고리별 분할
            n_eval = max(1, int(len(category_df) * test_size))
            category_eval = category_df.sample(n=n_eval, random_state=seed)
            category_train = category_df.drop(category_eval.index)
            
            train_dfs.append(category_train)
            eval_dfs.append(category_eval)
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        eval_df = pd.concat(eval_dfs, ignore_index=True)
        
        # 셔플
        train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        eval_df = eval_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        return train_df, eval_df
    
    def _simple_split(self, df, test_size, seed):
        """간단한 데이터 분할"""
        eval_df = df.sample(frac=test_size, random_state=seed)
        train_df = df.drop(eval_df.index)
        return train_df, eval_df
    
    def create_category_datasets(self, df):
        """
        카테고리별 개별 데이터셋 생성
        
        Args:
            df: 원본 데이터프레임
        
        Returns:
            dict: 카테고리별 Dataset 딕셔너리
        """
        if 'category' not in df.columns:
            return {"all": self.to_conversational_format(df)}
        
        category_datasets = {}
        
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            category_dataset = self.to_conversational_format(
                category_df, 
                include_category=True
            )
            category_datasets[category] = category_dataset
            print(f"{category} 데이터셋: {len(category_dataset)}개")
        
        return category_datasets
    
    def validate_format(self, dataset, format_type="conversational"):
        """
        데이터 형식 유효성 검사
        
        Args:
            dataset: Dataset 객체
            format_type: 형식 타입
        
        Returns:
            bool: 유효성 여부
        """
        try:
            sample = dataset[0]
            
            if format_type == "conversational":
                assert "messages" in sample
                assert isinstance(sample["messages"], list)
                assert len(sample["messages"]) >= 2
                
                for message in sample["messages"]:
                    assert "role" in message
                    assert "content" in message
                    assert message["role"] in ["system", "user", "assistant"]
            
            elif format_type == "instruction":
                assert "prompt" in sample
                assert "completion" in sample
            
            elif format_type == "custom":
                assert "text" in sample
            
            print(f"✅ {format_type} 형식 검증 성공")
            return True
            
        except Exception as e:
            print(f"❌ {format_type} 형식 검증 실패: {e}")
            return False

# 사용 예시
if __name__ == "__main__":
    # 예시 데이터프레임
    import pandas as pd
    
    sample_data = {
        'question': [
            "전세 계약에서 임대인이 보증금을 돌려주지 않는다면?",
            "근저당권의 피담보채권 범위는?"
        ],
        'answer': [
            "민법상 채무불이행에 해당하며 내용증명 후 소송 가능합니다.",
            "원본, 이자, 위약금, 손해배상금이 포함됩니다."
        ],
        'category': ["전세사기", "부동산물권"],
        'difficulty': ["중급", "중급"]
    }
    
    df = pd.DataFrame(sample_data)
    
    # 포맷터 초기화
    formatter = DataFormatter()
    
    # Conversational 형식 변환
    conv_dataset = formatter.to_conversational_format(df)
    
    # 형식 검증
    formatter.validate_format(conv_dataset, "conversational")
    
    # 훈련/검증 분할
    train_dataset, eval_dataset = formatter.split_train_eval(conv_dataset)
    
    print(f"훈련 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(eval_dataset)}개")
