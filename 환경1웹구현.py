import streamlit as st
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 기존 클래스 및 함수들 import
from content_based import ProductRecommendationSystem as ContentBasedRS
from collaborative import ProductRecommendationSystem as CollaborativeRS



def load_item_profiles(json_path):
    """JSON 파일에서 item_profiles를 불러옵니다."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            item_profiles = json.load(f)
        return item_profiles
    except Exception as e:
        st.error(f"item_profiles 로드 중 오류 발생: {str(e)}")
        return None

def get_category_path_string(categories):
    """카테고리 리스트를 문자열로 변환합니다."""
    return ' > '.join(categories) if categories else ''

def load_data():
    """데이터 로드 함수"""
    try:
        item_profiles = load_item_profiles('item_profiles_with_embeddings.json')
        customers_df = pd.read_csv('customers_df.csv')
        orders_paths = ['order1_df.csv', 'order2_df.csv', 'order3_df.csv', 'order4_df.csv', 'order5_df.csv']
        return item_profiles, customers_df, orders_paths
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None, None, None

def get_customer_info(customers_df, customer_id):
    """고객 정보 조회"""
    customer_info = customers_df[customers_df['고객식별ID'] == customer_id]
    if customer_info.empty:
        return None
    return customer_info.iloc[0]['성별'], customer_info.iloc[0]['연령대']

def find_most_frequent_items_by_demographics(orders_paths, customers_df, item_profiles, gender, age_group, number_filter, top_n=10):
    """인구통계학적 추천"""
    demographic_item_count = Counter()
    
    for orders_path in orders_paths:
        df = pd.read_csv(orders_path)
        merged_df = pd.merge(df, customers_df, on='고객식별ID')
        filtered_df = merged_df[
            (merged_df['성별'] == gender) & 
            (merged_df['연령대'] == age_group)
        ]
        
        for _, row in filtered_df.iterrows():
            item_id = str(row['상품ID'])
            if item_id in item_profiles:
                demographic_item_count[item_id] += 1
    
    recommendations = []
    for item_id, count in demographic_item_count.most_common():
        item_name = item_profiles[item_id].get('name', '')
        if item_name.endswith(str(number_filter)):
            item_categories = item_profiles[item_id].get('categories', [])
            category_path = ' > '.join(item_categories) if item_categories else 'Unknown'
            
            recommendations.append({
                'item_id': item_id,
                'name': item_name,
                'categories': category_path,
                'order_count': count
            })
            
            if len(recommendations) >= top_n:
                break
                
    return recommendations

def main():
    st.title("상품 추천(홈 화면)")
    
    # 데이터 로드
    item_profiles, customers_df, orders_paths = load_data()
    # if not all([item_profiles is not None, customers_df.any(), orders_paths]):
    #     st.error("필요한 데이터를 로드할 수 없습니다.")
    #     return
        
    # 사이드바에 입력 폼 생성
    with st.sidebar:
        st.header("검색 조건 입력")
        customer_id = st.text_input("고객 ID")
        number_filter = st.selectbox("호텔 번호", options=[1, 2, 3, 4, 5])
        
    if not customer_id:
        st.info("고객 ID를 입력해주세요.")
        return
        
    # 고객 정보 확인
    customer_info = get_customer_info(customers_df, customer_id)
    if not customer_info:
        st.error("존재하지 않는 고객 ID입니다.")
        return
        
    gender, age_group = customer_info
    # st.write(f"### 고객 정보")
    # st.write(f"성별: {gender}, 연령대: {age_group}")

    if gender == 'male':
        gender1 = '남성'
    else:
        gender1 = '여성'

    
    # 세 가지 추천 시스템 실행
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### 내가 구매한 상품과 유사한 상품")
        content_rs = ContentBasedRS()
        content_rs.create_category_graph("left_joined_file.csv")
        purchase_history, item_history = content_rs.create_user_purchase_history(
            orders_paths, item_profiles
        )
        
        if customer_id in purchase_history:
            recommendations = content_rs.recommend_products(
                purchase_history, item_history, item_profiles, 
                customer_id, number_filter=str(number_filter)
            )
            
            for rec in recommendations[:10]:
                with st.expander(rec['name']):
                    st.write(f"카테고리: {get_category_path_string(item_profiles[rec['item_id']]['categories'])}")
                    st.write(f"카테고리 유사도: {rec['category_similarity']:.4f}")
                    st.write(f"설명 유사도: {rec['description_similarity']:.4f}")
            
            

    
    with col2:
        st.write("### 나와 유사한 사람들이 구매한 상품")
        collab_rs = CollaborativeRS()
        collab_rs.create_category_graph("left_joined_file.csv")
        purchase_history, item_history = collab_rs.create_user_purchase_history(
            orders_paths, item_profiles
        )
        
        if customer_id in purchase_history:
            recommendations = collab_rs.recommend_collaborative(
                purchase_history, item_history, item_profiles, 
                customer_id, number_filter=str(number_filter)
            )
            
            for rec in recommendations[:10]:
                with st.expander(rec['name']):
                    st.write(f"카테고리: {get_category_path_string(item_profiles[rec['item_id']]['categories'])}")
                    st.write(f"카테고리 유사도: {rec['category_similarity']:.4f}")
                    st.write(f"설명 유사도: {rec['description_similarity']:.4f}")
            
    
    with col3:
        st.write(f"### {age_group}대 {gender1}이 많이 구매한 상품")
        demographic_recommendations = find_most_frequent_items_by_demographics(
            orders_paths, customers_df, item_profiles,
            gender, age_group, number_filter
        )
        
        for rec in demographic_recommendations[:10]:
            with st.expander(rec['name']):
                st.write(f"카테고리: {rec['categories']}")
                st.write(f"주문 횟수: {rec['order_count']}")

        

if __name__ == "__main__":
    main()
