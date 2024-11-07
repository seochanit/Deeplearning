from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# 데이터 전처리 클래스 정의
class PreprocessData:
    def __init__(self, valid_size, random_state, scaling=False):
        self.valid_size = valid_size
        self.random_state = random_state
        self.scaling = scaling

    def load_datasets(self):
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        return train_images, train_labels, test_images, test_labels

    def scaled_pixels(self, images, labels):
        if self.scaling:
            images = images / 255.0
        return images.astype('float32'), to_categorical(labels, 10)

    def preprocess_data(self):
        train_images, train_labels, test_images, test_labels = self.load_datasets()
        train_images, train_labels = self.scaled_pixels(train_images, train_labels)
        test_images, test_labels = self.scaled_pixels(test_images, test_labels)
        return train_images, train_labels, test_images, test_labels

# 데이터 로드 및 전처리
datasets = PreprocessData(valid_size=0.15, random_state=42, scaling=False)
train_images, train_labels, test_images, test_labels = datasets.preprocess_data()

# 데이터 증강 생성기 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow(train_images, train_labels, batch_size=128)
test_gen = test_datagen.flow(test_images, test_labels, batch_size=128)

# 사전 학습된 ResNet50 모델 로드 (ImageNet 가중치 사용)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# 새로운 출력층 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# 사전 학습된 모델의 레이어는 고정(freeze)
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
rlr_call = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=3, verbose=1)
es_call = EarlyStopping(monitor='val_loss', mode='min', patience=6, verbose=1)

# 첫 번째 학습 단계: 사전 학습된 레이어는 고정한 상태에서 상위 레이어만 학습
print("Initial training with frozen base model layers...")
history = model.fit(train_gen, epochs=20, validation_data=test_gen, callbacks=[rlr_call, es_call])

# fine-tuning: 사전 학습된 모델의 상위 10개 레이어를 학습 가능하게 설정
for layer in base_model.layers[-10:]:
    layer.trainable = True

# 모델 재컴파일 (fine-tuning 학습률 조정)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# fine-tuning 단계 추가 훈련
print("Fine-tuning with partially unfrozen base model layers...")
history_finetune = model.fit(train_gen, epochs=30, validation_data=test_gen, callbacks=[rlr_call, es_call])

# 모델 평가
loss, accuracy = model.evaluate(test_gen)
print(f'Test accuracy: {accuracy:.4f}')

# 학습 기록 그래프
def plot_history(histories):
    plt.figure(figsize=(12, 4))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(histories[0].history['loss'], label='Initial Training Loss')
    plt.plot(histories[0].history['val_loss'], label='Initial Validation Loss')
    plt.plot(histories[1].history['loss'], label='Fine-tuning Loss')
    plt.plot(histories[1].history['val_loss'], label='Fine-tuning Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(histories[0].history['accuracy'], label='Initial Training Accuracy')
    plt.plot(histories[0].history['val_accuracy'], label='Initial Validation Accuracy')
    plt.plot(histories[1].history['accuracy'], label='Fine-tuning Accuracy')
    plt.plot(histories[1].history['val_accuracy'], label='Fine-tuning Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history([history, history_finetune])
