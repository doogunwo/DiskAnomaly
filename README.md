# DiskAnomaly

스토리지(디스크)는 호스트 입장에서 블랙박스 구조로 동작하는 컴포넌트입니다.
그러나 스토리지는 현대 컴퓨팅의 가장 기본적인 요소입니다. 

하지만 기존 디스크 트레이스 툴로는 디스크 이상 탐지에 한계가 존재합니다.
![image](https://github.com/user-attachments/assets/bf7787ea-69b0-4c9c-9dc9-104803a80344)
![image](https://github.com/user-attachments/assets/0be392a3-00fc-4af6-945a-8a4d86158250)

위 화면으로는 시각적으로 디스크의 이상 탐지를 쉽게 확인하기 어렵습니다. 
따라서 이러한 데이터를 기반으로 디스크 이상 탐지를 실시간으로 수행하는 모델을 연구 및 개발했습니다.

사용하는 데이터 필드 -> Timestamp, I/O Type, Size, Sector
![image](https://github.com/user-attachments/assets/2717c17b-449f-4801-b156-93c6c3c5adb7)

모델 레이어는 위와 같이 구현했는데요, 위 구조는 아래 논문의 아키텍처를 간략화한 구조입니다.
[1] A One-Class Anomaly Detection Method for Drives based on Adversarial Auto-Encoder, Yufei Wang, 2022 IEEE 24th Int Conf on High Performance Computing & Communications

학습은 A6000 그래픽 카드를 사용하였습니다. 학습 데이터 자체에서 테스트 샘플링을 추출하여 학습을 수행했기 때문에
Validation 데이터셋을 별도로 구성하지 않았습니다.

학습 데이터셋 수집은 디스크 트레이스툴을 실행하고, FIO를 활용하여 I/O를 발생시켰습니다.
학습 파라미터는 
입력 차원 4, 은닉 차원 64, 잠재 차원 32를 배치했고 MSELoss를 사용하였습니다.
옵티마이저로는 RMSprop를 사용했습니다. 
 
  ![image](https://github.com/user-attachments/assets/e12336e3-962a-48c0-8e03-def975211e3a)

테스트 결과는 위와 같습니다. 결론적으로 해당 실험에서 간과한 문제가 있었습니다.
어떤 기준점을 토대로 디스크 이상탐지를 해야 하는지 기준점을 찾지 못하여,
학습 데이터 자체에서 비이상과 이상을 구분하지 않았습니다.
이러한 문제로 모델에 대한 학습이 제대로 수행되었는지 평가가 어려우며, 실제 테스트 결과에서도 해당 결과가
비이상적인지 판단이 어려웠습니다. 
