from CNN import PreprocessData, CnnModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# 데이터 전처리 및 모델 초기화
datasets = PreprocessData(valid_size=0.1, random_state=42, scaling=False)  # valid_size=0.1로 설정하여 조건 충족
tr_images, tr_ohe_labels, val_images, val_ohe_labels, test_images, test_ohe_labels = datasets.preprocess_data()

model = CnnModel.create_model(verbose=True)

# 데이터 증강 생성기 설정
tr_gen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1/255.0,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=[0.8, 1.2]
)

val_gen = ImageDataGenerator(rescale=1/255.0)

# 데이터 생성기
flow_tr_gen = tr_gen.flow(x=tr_images, y=tr_ohe_labels, batch_size=128, shuffle=True)
flow_val_gen = val_gen.flow(x=val_images, y=val_ohe_labels, batch_size=128, shuffle=False)

# 콜백 설정
rlr_call = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=4, verbose=1)
es_call = EarlyStopping(monitor='val_loss', mode='min', patience=7, verbose=1)
checkpoint = ModelCheckpoint(
    'best_model.keras',           # 모델을 저장할 파일명
    monitor='val_loss',        # val_loss 기준으로 저장
    save_best_only=True,       # 가장 좋은 성능일 때만 저장
    mode='min',                # 손실이 최소일 때 저장
    verbose=1
)

# 콜백 리스트
callbacks = [rlr_call, es_call, checkpoint]

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
print("Training model with checkpoint saving...")
tr_hist = model.fit(flow_tr_gen, epochs=200, validation_data=flow_val_gen, callbacks=callbacks)

# 테스트 데이터 평가
test_gen = ImageDataGenerator(rescale=1/255.0)
flow_test_gen = test_gen.flow(x=test_images, y=test_ohe_labels, batch_size=32, shuffle=False)
test_hist = model.evaluate(flow_test_gen)

# 학습 기록 그래프
history = tr_hist.history

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
