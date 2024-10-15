# Deeplearning

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
