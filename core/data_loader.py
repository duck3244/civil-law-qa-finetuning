# data_loader.py
# 데이터 로딩 및 전처리 담당

import pandas as pd
from datasets import Dataset
import json

class CivilLawDataLoader:
    """민법 QA 데이터셋 로더"""
    
    def __init__(self, filename="civil_law_qa_dataset.csv"):
        self.filename = filename
        self.df = None
        self.stats = {}
    
    def load_dataset(self):
        """CSV 파일 로드 및 기본 분석"""
        try:
            # 다양한 인코딩으로 시도
            encodings = ['utf-8', 'cp949', 'euc-kr']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.filename, encoding=encoding)
                    print(f"파일 로드 성공 (인코딩: {encoding})")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                raise Exception("모든 인코딩 시도 실패")
            
            self._analyze_dataset()
            return self.df
            
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {self.filename}")
            return None
        except Exception as e:
            print(f"파일 로드 중 오류: {e}")
            return None
    
    def _analyze_dataset(self):
        """데이터셋 구조 분석"""
        print(f"데이터셋 로드 완료: {len(self.df)}개 샘플")
        print(f"컬럼: {self.df.columns.tolist()}")
        
        # 통계 정보 저장
        self.stats = {
            'total_samples': len(self.df),
            'columns': self.df.columns.tolist()
        }
        
        # 카테고리 분포
        if 'category' in self.df.columns:
            category_dist = self.df['category'].value_counts()
            self.stats['categories'] = category_dist.to_dict()
            print(f"\n카테고리별 분포:")
            print(category_dist)
        
        # 난이도 분포
        if 'difficulty' in self.df.columns:
            difficulty_dist = self.df['difficulty'].value_counts()
            self.stats['difficulties'] = difficulty_dist.to_dict()
            print(f"\n난이도별 분포:")
            print(difficulty_dist)
        
        # 텍스트 길이 분석
        self._analyze_text_length()
        
        # 샘플 데이터 출력
        self._show_samples()
    
    def _analyze_text_length(self):
        """질문/답변 길이 분석"""
        if 'question' in self.df.columns and 'answer' in self.df.columns:
            self.df['question_length'] = self.df['question'].str.len()
            self.df['answer_length'] = self.df['answer'].str.len()
            
            q_stats = {
                'mean': float(self.df['question_length'].mean()),
                'max': int(self.df['question_length'].max()),
                'min': int(self.df['question_length'].min())
            }
            
            a_stats = {
                'mean': float(self.df['answer_length'].mean()),
                'max': int(self.df['answer_length'].max()),
                'min': int(self.df['answer_length'].min())
            }
            
            self.stats['question_length'] = q_stats
            self.stats['answer_length'] = a_stats
            
            print(f"\n질문 길이 통계:")
            print(f"평균: {q_stats['mean']:.1f}자, 최대: {q_stats['max']}자, 최소: {q_stats['min']}자")
            
            print(f"\n답변 길이 통계:")
            print(f"평균: {a_stats['mean']:.1f}자, 최대: {a_stats['max']}자, 최소: {a_stats['min']}자")
    
    def _show_samples(self, num_samples=3):
        """샘플 데이터 출력"""
        print(f"\n=== 샘플 데이터 (상위 {num_samples}개) ===")
        for i in range(min(num_samples, len(self.df))):
            print(f"\n[샘플 {i+1}]")
            print(f"질문: {self.df.iloc[i]['question']}")
            print(f"답변: {self.df.iloc[i]['answer'][:100]}...")
            
            if 'category' in self.df.columns:
                print(f"카테고리: {self.df.iloc[i]['category']}")
            if 'difficulty' in self.df.columns:
                print(f"난이도: {self.df.iloc[i]['difficulty']}")
    
    def preprocess_data(self):
        """데이터 전처리"""
        if self.df is None:
            print("데이터가 로드되지 않았습니다.")
            return None
        
        print("\n=== 데이터 전처리 시작 ===")
        original_count = len(self.df)
        
        # 필수 컬럼 확인
        if 'question' not in self.df.columns or 'answer' not in self.df.columns:
            print("오류: 'question' 또는 'answer' 컬럼이 없습니다.")
            return None
        
        # 빈 값 제거
        self.df = self.df.dropna(subset=['question', 'answer'])
        print(f"빈 값 제거: {original_count} → {len(self.df)}")
        
        # 빈 문자열 제거
        self.df = self.df[(self.df['question'].str.strip() != '') & 
                         (self.df['answer'].str.strip() != '')]
        print(f"빈 문자열 제거: {original_count} → {len(self.df)}")
        
        # 중복 질문 제거
        self.df = self.df.drop_duplicates(subset=['question'])
        print(f"중복 제거 후: {len(self.df)}")
        
        # 텍스트 정리
        self.df['question'] = self.df['question'].str.strip()
        self.df['answer'] = self.df['answer'].str.strip()
        
        return self.df
    
    def filter_by_category(self, categories=None):
        """카테고리별 필터링"""
        if self.df is None or 'category' not in self.df.columns:
            return self.df
        
        if categories is None:
            return self.df
        
        if isinstance(categories, str):
            categories = [categories]
        
        filtered_df = self.df[self.df['category'].isin(categories)]
        print(f"카테고리 필터링 ({categories}): {len(self.df)} → {len(filtered_df)}")
        
        return filtered_df
    
    def filter_by_difficulty(self, difficulties=None):
        """난이도별 필터링"""
        if self.df is None or 'difficulty' not in self.df.columns:
            return self.df
        
        if difficulties is None:
            return self.df
        
        if isinstance(difficulties, str):
            difficulties = [difficulties]
        
        filtered_df = self.df[self.df['difficulty'].isin(difficulties)]
        print(f"난이도 필터링 ({difficulties}): {len(self.df)} → {len(filtered_df)}")
        
        return filtered_df
    
    def split_by_category(self):
        """카테고리별 데이터 분할"""
        if self.df is None or 'category' not in self.df.columns:
            return {'all': self.df}
        
        category_splits = {}
        for category in self.df['category'].unique():
            category_df = self.df[self.df['category'] == category]
            category_splits[category] = category_df
            print(f"{category}: {len(category_df)}개")
        
        return category_splits
    
    def get_stats(self):
        """통계 정보 반환"""
        return self.stats
    
    def save_stats(self, output_path):
        """통계 정보를 JSON 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        print(f"통계 정보 저장: {output_path}")

# 사용 예시
if __name__ == "__main__":
    # 데이터 로더 초기화
    loader = CivilLawDataLoader("civil_law_qa_dataset.csv")
    
    # 데이터 로드
    df = loader.load_dataset()
    
    if df is not None:
        # 전처리
        processed_df = loader.preprocess_data()
        
        # 카테고리별 분할
        category_splits = loader.split_by_category()
        
        # 통계 저장
        loader.save_stats("dataset_stats.json")
