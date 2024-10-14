from CNN import PreprocessData, CnnModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

datasets = PreprocessData(valid_size=0.15, random_state=42, scaling=False)
tr_images, tr_ohe_labels, val_images, val_ohe_labels, test_images, test_ohe_labels = datasets.preprocess_data()

model = CnnModel.create_model(verbose=True)

tr_gen = ImageDataGenerator(horizontal_flip=True,
                            vertical_flip=True,
                            rescale=1/255.0,
                            rotation_range=0.45,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=[0.5,1.5])

val_gen = ImageDataGenerator(rescale=1/255.0)

flow_tr_gen = tr_gen.flow(x=tr_images, y=tr_ohe_labels, batch_size=64, shuffle=True)
flow_val_gen = val_gen.flow(x=val_images, y=val_ohe_labels, batch_size=64, shuffle=False)

rlr_call = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=4, verbose=1)
es_call = EarlyStopping(monitor='val_loss', mode='min', patience=7, verbose=1)

model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

tr_hist = model.fit(flow_tr_gen, epochs=200, validation_data=flow_val_gen, callbacks=[rlr_call, es_call])

test_gen = ImageDataGenerator(rescale=1/255.0)
flow_test_gen = test_gen.flow(x=test_images, y=test_ohe_labels, batch_size=32, shuffle=False)
test_hist = model.evaluate(flow_test_gen)

import matplotlib.pyplot as plt

# 모델 학습 후 얻은 기록
history = tr_hist.history

# 손실(loss) 그래프 그리기
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 정확도(accuracy) 그래프 그리기
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 그래프 보여주기
plt.tight_layout()
plt.show()
