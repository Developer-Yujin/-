import streamlit as st
import pandas as pd
from collections import defaultdict
import json
from content_based import ProductRecommendationSystem

# 기존 함수들을 그대로 유지
def get_category_path_string(categories):
    return ' > '.join(categories) if categories else 'Unknown'

def load_item_profiles(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        item_profiles = json.load(f)
    return item_profiles

def find_frequently_bought_together(orders_paths, item_profiles, target_item_id, number_filter, top_n=10):
    item_count = defaultdict(int)
    
    target_item_name = item_profiles.get(target_item_id, {}).get('name', 'Unknown')
    target_item_categories = item_profiles.get(target_item_id, {}).get('categories', [])
    target_category_path = get_category_path_string(target_item_categories)
    
    for orders_path in orders_paths:
        df = pd.read_csv(orders_path)
        grouped_orders = df.groupby('주문번호')['상품ID'].apply(list)
        
        for order in grouped_orders:
            if target_item_id in order:
                for item_id in order:
                    if item_id != target_item_id:
                        item_count[item_id] += 1
    
    sorted_items = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
    results = []
    
    for item_id, count in sorted_items:
        if item_id in item_profiles:
            item_name = item_profiles[item_id].get('name', 'Unknown')
            if item_name.endswith(str(number_filter)):
                item_categories = item_profiles[item_id].get('categories', [])
                category_path = get_category_path_string(item_categories)
                results.append({
                    'item_id': item_id,
                    'name': item_name,
                    'categories': category_path,
                    'count': count
                })
        if len(results) >= top_n:
            break
    
    return results, target_item_name, target_category_path

def recommend_similar_items(recommender, item_profiles, target_item_id, ending_numbers, top_n=10):
    recommendations = []
    
    target_item = item_profiles.get(target_item_id)
    if not target_item:
        return [], None, None

    target_item_name = target_item.get('name')
    target_item_categories = target_item.get('categories')
    target_item_embedding = target_item.get('embedding', None)
    target_category_path = get_category_path_string(target_item_categories)

    if not target_item_categories or not target_item_name:
        return [], None, None

    for item_id, item_data in item_profiles.items():
        if item_id == target_item_id:
            continue

        item_name = item_data.get('name')
        item_categories = item_data.get('categories')
        item_embedding = item_data.get('embedding', None)

        if not item_name or not item_categories:
            continue

        if not any(item_name.endswith(str(num)) for num in ending_numbers):
            continue

        max_category_similarity = 0
        if all(node in recommender.graph for node in target_item_categories) and \
           all(node in recommender.graph for node in item_categories):
            category_similarity = recommender.wu_palmer_similarity(target_item_categories, item_categories)
            max_category_similarity = max(max_category_similarity, category_similarity)

        max_description_similarity = 0
        if target_item_embedding and item_embedding:
            description_similarity = recommender.compute_description_similarity(target_item_embedding, item_embedding)
            max_description_similarity = max(max_description_similarity, description_similarity)

        if max_category_similarity > 0:
            recommendations.append({
                'item_id': item_id,
                'name': item_name,
                'categories': get_category_path_string(item_categories),
                'category_similarity': max_category_similarity,
                'description_similarity': max_description_similarity
            })

    recommendations.sort(key=lambda x: (x['category_similarity'], x['description_similarity']), reverse=True)
    return recommendations[:top_n], target_item_name, target_category_path

# Streamlit 앱 메인 함수
def main():
    st.title('추천 시스템(환경2)')
    

    # 사이드바에 입력 폼 배치
    with st.sidebar:
        st.header("검색 조건 입력")
        target_item_id = st.text_input('상품 ID')
        number_filter = st.selectbox("호텔 번호", options=[1, 2, 3, 4, 5])

    if target_item_id and number_filter:
        try:
            # 데이터 로드
            item_profiles = load_item_profiles('item_profiles_with_embeddings.json')
            recommender = ProductRecommendationSystem()
            recommender.create_category_graph("left_joined_file.csv")
            orders_paths = ['order1_df.csv', 'order2_df.csv', 'order3_df.csv', 
                          'order4_df.csv', 'order5_df.csv']

            # 두 가지 추천 결과 계산
            similar_items, target_name1, target_category1 = recommend_similar_items(
                recommender, item_profiles, target_item_id, [number_filter])
            bought_together, target_name2, target_category2 = find_frequently_bought_together(
                orders_paths, item_profiles, target_item_id, number_filter)

            # 타겟 상품 정보 표시
            st.header('상품 정보')
            st.write(f'상품명: {target_name1}')
            st.write(f'카테고리: {target_category1}')

            # 두 개의 컬럼으로 결과 표시
            col1, col2 = st.columns(2)

            with col1:
                st.header('유사한 상품')
                if similar_items:
                    for item in similar_items:
                        with st.expander(f"{item['name']}"):
                            st.write(f"카테고리: {item['categories']}")
                            st.write(f"카테고리 유사도: {item['category_similarity']:.4f}")
                            st.write(f"설명 유사도: {item['description_similarity']:.4f}")
                else:
                    st.write("추천 결과가 없습니다.")

            with col2:
                st.header('함께 구매된 상품')
                if bought_together:
                    for item in bought_together:
                        with st.expander(f"{item['name']}"):
                            st.write(f"카테고리: {item['categories']}")
                            st.write(f"함께 구매된 횟수: {item['count']}")
                else:
                    st.write("추천 결과가 없습니다.")

        except Exception as e:
            st.error(f'오류가 발생했습니다: {str(e)}')

if __name__ == '__main__':
    main()