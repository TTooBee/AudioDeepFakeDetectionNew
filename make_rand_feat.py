import numpy as np

# 16개의 랜덤 행렬을 생성하여 각 행렬을 txt 파일로 저장하는 함수
def create_and_save_random_matrices(num_matrices, rows, cols):
    for i in range(num_matrices):
        # 랜덤 행렬 생성
        random_matrix = np.random.rand(rows, cols)
        
        # 파일명 설정
        filename = f'random_matrix_{i+1}.txt'
        
        # 랜덤 행렬을 txt 파일로 저장
        np.savetxt(filename, random_matrix, fmt='%.6f')
        
        print(f"랜덤 행렬 {i+1}이 '{filename}' 파일로 저장되었습니다.")

# 함수 호출
create_and_save_random_matrices(16, 300, 12)
