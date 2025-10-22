from networks.codenet.codenet import CodeNet

import json
import os

if __name__ == "__main__":
	# 입력 데이터 로드
	input_dir = "inputs"
	with open(f"{input_dir}/data.json", "r", encoding="utf-8") as f:
		input_data = json.load(f)
		
	# 네트워크 그래프 컴파일 및 실행
	codenet = CodeNet() # CodeNet 객체 생성
	codenet.compile() # 그래프 로드 및 컴파일
	ne = codenet.get_nodes_edges()
	print(ne) # 노드 및 엣지 정보 출력
	
	# state = {'input_data': input_data} # 그래프에 전달할 초기 데이터
	result = codenet.run(state={'input_data': input_data}) # 초기 데이터와 함께 그래프 실행

	# 결과를 json 파일로 저장 (outputs 디렉토리가 없으면 생성)
	output_dir = "outputs"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	file_path = f"{output_dir}/output.json"
	with open(file_path, "w") as f:
		json.dump(result, f, indent=4, ensure_ascii=False)
	print(f"Output saved to {file_path}")
