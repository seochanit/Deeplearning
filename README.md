# Deeplearning

<<<<<<< HEAD
<코드 수정 사항 1104>
1. 배치 크기 조정
- 기존 batch_size=64에서 batch_size=128로 증가하여 학습 속도와 성능을 최적화.

2. 학습률 조정
- 학습률을 기존 0.01에서 0.001로 낮추어 모델의 안정적인 학습을 유도.
- Adam 옵티마이저의 기본 학습률을 낮추고, ReduceLROnPlateau 콜백을 통해 검증 손실이 개선되지 않을 경우 학습률을 점진적으로 줄임.

3. 데이터 증강 조정
- rotation_range와 zoom_range를 CIFAR-10 이미지 크기에 맞게 최적화:
    - rotation_range=30으로 줄여 과도한 이미지 회전을 방지.
    - zoom_range=[0.8, 1.2]로 줄여 이미지 축소와 확대 범위를 CIFAR-10 데이터에 맞게 최적화.

4. L2 정규화 추가
- 각 Conv2D 및 Dense 레이어에 kernel_regularizer=l2(0.001) 옵션을 추가하여 과적합을 방지.
- tensorflow.keras.regularizers에서 l2 함수를 추가로 임포트하여 L2 정규화를 활성화.
=======
cifar10 dataset으로 신경망 모델 학습
https://www.kaggle.com/datasets/oxcdcd/cifar10

* 목표 : test data 정확도 50% 이상

1. 데이터 불러오기 : ImageDataGenerator(train, test)
2. 데이터 전처리 : /255
3. 신경망모델 라이브러리 블러오기
4. 신경망 모델 선정 : Dense, Dropout, BatchNormalization
5. compile 설정
6. fit(validation_split=0.1)
7. loss / val_loss graph
8. accuracy / val_acc graph
9. test data evaluate
10. 모델 설정 및 epochs/batch_size 조절 (epochs<=300 / batch_size<=300)
>>>>>>> 45dab0d735b1c597d8e757d704073e932728632d
