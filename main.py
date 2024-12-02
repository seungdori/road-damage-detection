import os
import numpy as np
import tensorflow as tf
from keras import layers, models, mixed_precision
from keras.api.applications import EfficientNetV2S
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.src.regularizers import L2 as l2
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import datetime
import gc
import numpy as np
import keras

import math

import time
# 데이터 경로 설정

IMG_SIZE = 224
BATCH_SIZE = 256
NUM_CLASSES = 7

#================================================================================================

class ImprovedTraining:
    def __init__(self, model_builder, train_generator, valid_generator,
                 batch_size=128, epochs=100, n_folds=5, initial_lr=1e-3, class_weights=None):
        self.model_builder = model_builder
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.batch_size = batch_size
        
        self.epochs = epochs
        self.n_folds = n_folds
        self.initial_lr = initial_lr
        self.learning_rate = initial_lr
        self.img_size = 224
        self.num_classes = 7
        
        # GPU 설정
        physical_devices = tf.config.list_physical_devices('GPU')
        self.use_mixed_precision = len(physical_devices) > 0
        
        if physical_devices:
            try:
                # A100에 맞는 메모리 설정
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                
                # A100은 40GB/80GB 메모리를 가지므로 높은 메모리 제한 설정 가능
                tf.config.experimental.set_virtual_device_configuration(
                    physical_devices[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=40960)]  # 40GB
                )
                
                # Mixed precision 설정 최적화
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                print("Mixed precision policy enabled:", policy.name)
                
                # XLA 최적화
                tf.config.optimizer.set_jit(True)
                
                # A100에 최적화된 성능 옵션
                tf.config.optimizer.set_experimental_options({
                    "layout_optimizer": True,
                    "constant_folding": True,
                    "shape_optimization": True,
                    "remapping": True,
                    "arithmetic_optimization": True,
                    "dependency_optimization": True,
                    "loop_optimization": True,
                    "function_optimization": True,
                    "debug_stripper": True,
                    "scoped_allocator_optimization": True,
                    "pin_to_host_optimization": True,
                    "implementation_selector": True,
                    "auto_mixed_precision": True,
                    "min_graph_nodes": 1,
                })
                
            except Exception as e:
                print(f"GPU 설정 중 오류 발생: {str(e)}")
                self.use_mixed_precision = False
        
        # 데이터 로딩 최적화
        self.AUTOTUNE = tf.data.AUTOTUNE
        
        # A100에 최적화된 데이터셋 성능 옵션
        self.dataset_options = tf.data.Options()
        self.dataset_options.experimental_optimization.map_parallelization = True
        self.dataset_options.experimental_optimization.parallel_batch = True
        self.dataset_options.experimental_deterministic = False
        self.dataset_options.threading.private_threadpool_size = 16
        self.dataset_options.threading.max_intra_op_parallelism = 16
        
        tf.config.optimizer.set_jit(True)

    

    def train_final_model(self, use_custom_training=False):
        print("데이터셋 준비 중...")
        # 전체 데이터 크기 계산
        total_samples = len(self.train_generator)
        total_val_samples = len(self.valid_generator)
        batch_size = self.train_generator.batch_size
        
        # 데이터 로딩
        pbar = tqdm(total=total_samples, desc="Loading training samples")
        X_data_list = []
        y_data_list = []
        
        try:
            for i in range(0, total_samples, batch_size):
                try:
                    batch_x, batch_y = next(iter(self.train_generator))
                    if tf.is_tensor(batch_x):
                        batch_x = batch_x.numpy()
                    if tf.is_tensor(batch_y):
                        batch_y = batch_y.numpy()
                    
                    X_data_list.append(batch_x)
                    y_data_list.append(batch_y)
                    pbar.update(batch_size)
                except StopIteration:
                    print("\nData generator exhausted")
                    break
                except Exception as e:
                    print(f"\nError loading batch: {str(e)}")
                    continue
                
                if len(X_data_list) % 10 == 0:
                    gc.collect()
        finally:
            pbar.close()
            
        # 검증 데이터 로딩
        pbar = tqdm(total=total_val_samples, desc="Loading validation samples")
        X_val_list = []
        y_val_list = []
        
        try:
            for i in range(0, total_val_samples, batch_size):
                try:
                    batch_x, batch_y = next(iter(self.valid_generator))
                    if tf.is_tensor(batch_x):
                        batch_x = batch_x.numpy()
                    if tf.is_tensor(batch_y):
                        batch_y = batch_y.numpy()
                    
                    X_val_list.append(batch_x)
                    y_val_list.append(batch_y)
                    pbar.update(batch_size)
                except StopIteration:
                    print("\nValidation data generator exhausted")
                    break
                except Exception as e:
                    print(f"\nError loading validation batch: {str(e)}")
                    continue
        finally:
            pbar.close()
            
        print("\n데이터 연결 중...")
        X_data = np.concatenate(X_data_list, axis=0)
        y_data = np.concatenate(y_data_list, axis=0)
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        
        print(f"로드된 데이터 형태: X={X_data.shape}, y={y_data.shape}")
        print(f"검증 데이터 형태: X_val={X_val.shape}, y_val={y_val.shape}")
        
        # 메모리 정리
        del X_data_list, y_data_list, X_val_list, y_val_list
        gc.collect()
        
        print('\n최종 모델 학습 중...')
        
        # 데이터를 텐서로 변환
        X_data = tf.convert_to_tensor(X_data, dtype=tf.float32)
        y_data = tf.convert_to_tensor(y_data, dtype=tf.float32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
        
        with tf.device('/GPU:0'):
            input_shape = (self.img_size, self.img_size, 3)
            inputs = tf.keras.Input(shape=input_shape, name='input_layer')
            base_model = self.model_builder(inputs=inputs)
            
            if not isinstance(base_model, tf.keras.Model):
                outputs = base_model
                model = tf.keras.Model(inputs=inputs, outputs=outputs, name='classification_model')
            else:
                model = base_model
            
            # 데이터셋 생성
            train_dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data))
            train_dataset = train_dataset.shuffle(1000).batch(self.batch_size)
            
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_dataset = val_dataset.batch(self.batch_size)
            
            # 옵티마이저와 손실 함수 정의
            optimizer = tf.keras.optimizers.Adam(learning_rate=float(self.initial_lr))
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            
            # 학습 히스토리
            history = {
                'loss': [],
                'accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            }
            
            # Early stopping 변수
            best_val_loss = float('inf')
            patience_count = 0
            
            # 트레이닝 스텝 정의
            @tf.function
            def train_step(x, y):
                with tf.GradientTape() as tape:
                    logits = model(x, training=True)
                    loss = loss_fn(y, logits)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.argmax(logits, axis=1),
                            tf.argmax(y, axis=1)
                        ),
                        tf.float32
                    )
                )
                
                return loss, accuracy
            
            # Validation 스텝 정의
            @tf.function
            def val_step(x, y):
                logits = model(x, training=False)
                loss = loss_fn(y, logits)
                
                accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.argmax(logits, axis=1),
                            tf.argmax(y, axis=1)
                        ),
                        tf.float32
                    )
                )
                
                return loss, accuracy
            
            # 학습 루프
            for epoch in range(self.epochs):
                print(f'\nEpoch {epoch + 1}/{self.epochs}')
                
                # 트레이닝
                train_losses = []
                train_accuracies = []
                
                for x_batch, y_batch in train_dataset:
                    loss, acc = train_step(x_batch, y_batch)
                    train_losses.append(float(loss))
                    train_accuracies.append(float(acc))
                
                # Validation
                val_losses = []
                val_accuracies = []
                
                for x_batch, y_batch in val_dataset:
                    val_loss, val_acc = val_step(x_batch, y_batch)
                    val_losses.append(float(val_loss))
                    val_accuracies.append(float(val_acc))
                
                # 에포크 평균 계산
                train_loss = np.mean(train_losses)
                train_acc = np.mean(train_accuracies)
                val_loss = np.mean(val_losses)
                val_acc = np.mean(val_accuracies)
                
                # 히스토리 업데이트
                history['loss'].append(train_loss)
                history['accuracy'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                # 결과 출력
                print(
                    f'loss: {train_loss:.4f} - '
                    f'accuracy: {train_acc:.4f} - '
                    f'val_loss: {val_loss:.4f} - '
                    f'val_accuracy: {val_acc:.4f}'
                )
                
                # Early stopping 체크
                min_delta = 1e-4  # 최소 개선 기준
                if val_loss < (best_val_loss - min_delta):  # 의미있는 개선이 있을 때만 patience 리셋
                    best_val_loss = val_loss
                    model.save('best_model.keras')
                    patience_count = 0
                else:
                    patience_count += 1

                if patience_count >= 15:  # patience를 15로 증가
                    print('\nEarly stopping triggered')
                    break
        
        return model, history


class RoadDamageClassifier:
    #def __init__(self, train_dir, test_dir, img_size=416,batch_size = 64, num_classes=7): #<--원래의 코드. 
    def __init__(self, train_dir, test_dir, valid_dir, img_size=224, batch_size=32, 
                learning_rate=1e-3, epochs=50):
        """
        도로 손상 분류를 위한 통합 클래스 초기화
        
        Parameters:
        - train_dir: 학습 데이터 디렉토리 경로
        - test_dir: 테스트 데이터 디렉토리 경로
        - img_size: 입력 이미지 크기 (기본값: 224로 축소)
        - batch_size: 배치 크기 (기본값: 128로 증가)
        - num_classes: 분류할 클래스 수
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.valid_dir = valid_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = 7
        self.class_names = self._get_class_names()
        self.model = None
        self.callbacks = self._create_callbacks()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_save_dir = 'saved_models'
    def _create_callbacks(self):
        """콜백 함수 생성"""
        checkpoint = ModelCheckpoint(
            'best_model.keras', 
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
        
        return [checkpoint, early_stopping, reduce_lr]

    def _get_class_names(self):
        """클래스 이름 추출"""
        return sorted([d for d in os.listdir(self.train_dir) 
                      if os.path.isdir(os.path.join(self.train_dir, d))])
    
    def build_model(self):
        # 데이터 제너레이터 생성
        self.train_generator, self.valid_generator, self.test_generator = self.create_data_generators()
        self.num_classes = len(self.train_generator.class_indices)
        
        print("EfficientNetV2 기반 모델 생성 중...")
        self.model = build_improved_model(
            input_shape=(self.img_size, self.img_size, 3),
            num_classes=self.num_classes
        )
        print("모델 생성 완료")
        
        compile_start_time = time.time()
        
        # Learning Rate Scheduler 설정
        initial_learning_rate = self.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            first_decay_steps=500,     # 1000에서 500으로 감소
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1
        )
        
        # 컴파일 설정 개선
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            ),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.F1Score(name='f1_score', average='macro')
            ]
        )
        
        compile_end_time = time.time()
        print(f"모델 컴파일 완료: {compile_end_time - compile_start_time:.4f}초 소요")
        
        # 모델 구조 저장 (가독성 개선)
        model_structure_path = os.path.join(self.model_save_dir, 'model_structure.json')
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        with open(model_structure_path, 'w') as f:
            model_json = self.model.to_json()
            json.dump(json.loads(model_json), f, indent=2)  # 보기 좋게 정렬
        
        print(f"모델 구조 저장 완료: {model_structure_path}")
        
        
        return self.model
    def save_model(self, filepath, save_format='keras'):
        """모델 저장 함수"""
        # 모델 저장
        self.model.save(filepath, save_format=save_format)
        
        # 모델 정보 저장
        model_info = {
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'class_indices': self.train_generator.class_indices if hasattr(self, 'train_generator') else None,
            'training_config': self.model.get_config()
        }
        
        # 모델 정보를 JSON 파일로 저장
        info_filepath = filepath.rsplit('.', 1)[0] + '_info.json'
        with open(info_filepath, 'w') as f:
            json.dump(model_info, f, indent=4)
            print(f"모델 정보 저장 완료: {info_filepath}")

    def train(self, epochs=10):
        """모델 학습 실행"""
 
        # 데이터 준비
        
        start_time = time.time()
        print("데이터 준비 중...")
        train_generator, valid_generator, _ = self.create_data_generators()
        end_time = time.time()

        steps_per_epoch = len(train_generator.filenames) // self.batch_size
        validation_steps = len(valid_generator.filenames) // self.batch_size
        print(f"데이터 준비 완료: {end_time - start_time:.4f}초 소요")
        class_counts = np.bincount(train_generator.classes)
        total_samples = np.sum(class_counts)
        class_weights = {i: total_samples / (len(class_counts) * count) 
                        for i, count in enumerate(class_counts)}

        start_time = time.time()
        trainer = ImprovedTraining(
            model_builder=self.build_model(),
            train_generator=train_generator,
            valid_generator=valid_generator,

            batch_size=self.batch_size,
            epochs=self.epochs,
            n_folds=5,
            initial_lr=self.learning_rate,
            class_weights=class_weights  # 클래스 가중치 추가
        )
        end_time = time.time()
        print(f"학습 프로세스 초기화 완료: {end_time - start_time:.4f}초 소요")
        #================================================================================================
        start_time = time.time()
        # 최종 모델 학습
        final_model, history = trainer.train_final_model()
        end_time = time.time()
        print(f"최종 모델 학습 완료: {end_time - start_time:.4f}초 소요")
        #================================================================================================
        self.model = final_model
        return history
    

    

    
    def create_data_generators(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,          # 20에서 30으로 증가
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,           # 추가
            zoom_range=0.2,            # 추가
            vertical_flip=True,        # 추가
            brightness_range=[0.8,1.2], # 추가
            horizontal_flip=True,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,  # 배치 사이즈 확인
            class_mode='categorical',
            shuffle=True
        )

        valid_generator = test_datagen.flow_from_directory(
            self.valid_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,  # 배치 사이즈 확인
            class_mode='categorical'
        )

        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,  # 배치 사이즈 확인
            class_mode='categorical'
        )

        return train_generator, valid_generator, test_generator

    def compile_model(self, model, learning_rate=1e-3):
        """최적화된 모델 컴파일 설정"""
        def focal_loss(gamma=2., alpha=.25):
            def focal_loss_fixed(y_true, y_pred):
                epsilon = tf.keras.backend.epsilon()
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
                cross_entropy = -y_true * tf.math.log(y_pred)
                weight = alpha * tf.math.pow(1 - y_pred, gamma)
                loss = weight * cross_entropy
                return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
            return focal_loss_fixed

        # Mixed Precision 설정
        mixed_precision.set_global_policy('mixed_float16')
        
        # Learning Rate Schedule 설정
        initial_learning_rate = learning_rate
        decay_steps = 1000
        decay_rate = 0.9
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
        
        # Optimizer 설정
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        # 컴파일
        model.compile(
            optimizer=optimizer,
            loss=focal_loss(gamma=2.0, alpha=0.25),  # focal loss 적용
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC()
            ]
        )
        
        return model
    
    def create_training_callbacks(self):
        """학습 콜백 설정"""
        callbacks = [
            # 조기 종료 설정
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            ),
            
            # 학습률 조정 설정
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.1,
                patience=5,
                min_lr=1e-5
            ),
            
            # 모델 체크포인트 저장 - 파일 확장자를 .keras로 변경
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',  
                monitor='accuracy',   
                save_best_only=True
            )
        ]
        
        return callbacks
    
    def fine_tune_model(self, model, num_layers=20):
        """모델 미세조정 설정"""
        # EfficientNet의 마지막 일부 레이어 학습 가능하도록 설정
        base_model = model.layers[1]
        base_model.trainable = True
        
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
            
        return model

    def evaluate(self):
        """모델 평가 및 결과 시각화"""
        # Generator 생성
        _, _, test_generator = self.create_data_generators()
        
        # 테스트 세트가 비어있는지 확인
        if not test_generator.samples:
            print("경고: 테스트 데이터를 찾을 수 없습니다.")
            print(f"테스트 디렉토리 경로를 확인해주세요: {self.test_dir}")
            return
            
        print("\n예측 수행 중...")
        
        # 테스트 데이터를 배치로 로드하여 예측
        predictions_list = []
        y_true_list = []
        
        for i in range(len(test_generator)):
            x_batch, y_batch = next(iter(test_generator))
            batch_predictions = self.model.predict(x_batch, verbose=0)
            predictions_list.append(batch_predictions)
            y_true_list.extend(np.argmax(y_batch, axis=1))
            
        # 예측 결과 합치기
        predictions = np.vstack(predictions_list)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.array(y_true_list)
        
        # 분류 보고서 출력
        print("\n분류 보고서:")
        class_names = list(test_generator.class_indices.keys())
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # 혼동 행렬 계산 및 시각화
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()
        
        # 정확도 계산
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n테스트 정확도: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_true,
            'confusion_matrix': cm
        }
    def plot_training_history(self, history, save_path):
        """
        학습 히스토리를 시각화하고 저장하는 함수
        
        Args:
            history (dict): 학습 히스토리 딕셔너리
            save_path (str): 그래프를 저장할 경로
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss 그래프
        ax1.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:  # validation loss가 있는 경우에만 표시
            ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy 그래프
        ax2.plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:  # validation accuracy가 있는 경우에만 표시
            ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 그래프 저장
        if save_path:
            plt.savefig(save_path)
            print(f"학습 히스토리 그래프가 저장되었습니다: {save_path}")
        
        plt.show()
        plt.close()

#================================================================================================




def build_improved_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # 1. EfficientNetV2 백본
    base_model = EfficientNetV2S(
        include_top=False,
        weights='noisy-student',  # 사전 학습 가중치 변경
        input_tensor=inputs
    )
    
    # 2. 레이어 동결 설정
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    # 공간 정보 처리에 SeparableConv2D 적용
    fusion = layers.SeparableConv2D(256, 3, padding='same')(fusion)  # 효율적 컨볼루션
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Activation('relu')(fusion)

    # 3. 특징 추출
    layer_names = [layer.name for layer in base_model.layers if 'add' in layer.name]
    c3 = base_model.get_layer(layer_names[len(layer_names)//4]).output  # 14x14
    c4 = base_model.get_layer(layer_names[len(layer_names)//2]).output  # 14x14
    c5 = base_model.get_layer(layer_names[-1]).output  # 7x7
    
    print(f"Feature map sizes: C3:{c3.shape}, C4:{c4.shape}, C5:{c5.shape}")
    
    # 4. Attention 블록
    def attention_block(x):
        channel = x.shape[-1]
        global_avg = layers.GlobalAveragePooling2D()(x)
        global_avg = layers.Reshape((1, 1, channel))(global_avg)
        global_avg = layers.Conv2D(channel//8, 1)(global_avg)
        global_avg = layers.Activation('relu')(global_avg)
        global_avg = layers.Conv2D(channel, 1)(global_avg)
        global_avg = layers.Activation('sigmoid')(global_avg)
        return layers.Multiply()([x, global_avg])
    
    # 5. 수정된 특징 처리
    # C3, C4를 7x7 크기로 조정
    p3 = layers.Conv2D(256, 1, padding='same')(c3)
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(p3)  # 14x14 -> 7x7
    
    p4 = layers.Conv2D(256, 1, padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(p4)  # 14x14 -> 7x7
    
    p5 = layers.Conv2D(256, 1, padding='same')(c5)  # 이미 7x7
    
    # 특징 융합
    fusion = layers.Add()([p3, p4, p5])
    
    # 융합된 특징 개선
    fusion = layers.Conv2D(256, 3, padding='same')(fusion)
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Activation('relu')(fusion)
    
    # Attention 적용
    fusion = attention_block(fusion)
    
    # 6. 글로벌 평균 풀링
    x = layers.GlobalAveragePooling2D()(fusion)
    
    # 7. 분류 헤드
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # 드롭아웃 비율 0.4로 증가
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # 드롭아웃 비율 0.4로 증가
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)




#================================================================================================


# 이미지 전처리를 위한 별도 함수
def preprocess_images(source_dir, target_dir):
    """이미지 전처리 함수"""
    if not os.path.exists(source_dir):
        os.makedirs(source_dir, exist_ok=True)

    os.makedirs(target_dir, exist_ok=True)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        processed_class_path = os.path.join(target_dir, class_name)
        os.makedirs(processed_class_path, exist_ok=True)

        print(f"클래스 {class_name} 전처리 중...")
        for img_name in tqdm(os.listdir(class_path)):
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            enhanced = apply_clahe_enhancement(img)
            output_path = os.path.join(processed_class_path, img_name)
            cv2.imwrite(output_path, enhanced)

def build_improved_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # 1. EfficientNetV2 백본
    base_model = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    
    # 2. 레이어 동결 설정
    for layer in base_model.layers:
        layer.trainable = False
    trainable_layers = 50
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True
    
    # 3. 특징 추출
    layer_names = [layer.name for layer in base_model.layers if 'add' in layer.name]
    c3 = base_model.get_layer(layer_names[len(layer_names)//4]).output
    c4 = base_model.get_layer(layer_names[len(layer_names)//2]).output
    c5 = base_model.get_layer(layer_names[-1]).output
    
    print(f"Feature map sizes: C3:{c3.shape}, C4:{c4.shape}, C5:{c5.shape}")
    
    # 4. 채널 수 통일을 위한 1x1 컨볼루션
    p3 = layers.Conv2D(256, 1, padding='same')(c3)  # 14x14x256
    p4 = layers.Conv2D(256, 1, padding='same')(c4)  # 14x14x256
    p5 = layers.Conv2D(256, 1, padding='same')(c5)  # 7x7x256

    # 5. 특징맵 크기 조정 (모두 7x7로)
    p3 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(p3)  # 14x14 -> 7x7
    p4 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(p4)  # 14x14 -> 7x7
    # p5는 이미 7x7

    # 6. 특징 융합
    fusion = layers.Concatenate()([p3, p4, p5])  # Add 대신 Concatenate 사용
    
    # 7. 융합된 특징 처리
    fusion = layers.Conv2D(256, 1, padding='same')(fusion)  # 채널 수 감소
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Activation('relu')(fusion)
    
    # 8. 공간 정보 처리
    fusion = layers.Conv2D(256, 3, padding='same')(fusion)
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Activation('relu')(fusion)
    
    # 9. 글로벌 평균 풀링
    x = layers.GlobalAveragePooling2D()(fusion)
    
    # 10. 분류 헤드
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)
# Learning Rate Warmup 스케줄러
def get_lr_scheduler(initial_learning_rate=1e-4, warmup_epochs=5):
    def warmup_scheduler(epoch):
        if epoch < warmup_epochs:
            return initial_learning_rate * (epoch + 1) / warmup_epochs
        return initial_learning_rate
    
    return keras.callbacks.LearningRateScheduler(warmup_scheduler)

# 모델 컴파일 및 학습을 위한 설정
def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                keras.metrics.F1Score()]
    )
    return model


def apply_clahe_enhancement(img):
    """CLAHE 향상 기법 적용 함수"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)



def main(preprocess_data=True, use_pretrained=False, pretrained_model_path=None,
         batch_size=32, epochs=50, img_size=224, learning_rate=1e-3, output_dir='./models'):
    """
    메인 실행 함수
    매개변수:
        preprocess_data (bool): 데이터 전처리 수행 여부
        use_pretrained (bool): 사전 학습된 모델 사용 여부
        pretrained_model_path (str): 사전 학습된 모델 파일 경로
        batch_size (int): 배치 크기
        epochs (int): 학습 에포크 수
        img_size (int): 입력 이미지 크기
        learning_rate (float): 학습률
        output_dir (str): 결과 저장 디렉토리
    """
    # GPU 설정
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPU 사용 가능 상태: {tf.test.is_gpu_available()}")
    
    base_dir = os.getcwd()
    # 경로 설정
    TRAIN_DIR = os.path.join(base_dir, 'data/train')
    TEST_DIR = os.path.join(base_dir, 'data/test')
    VALID_DIR = os.path.join(base_dir, 'data/valid')
    PROCESSED_TRAIN_DIR = os.path.join(base_dir, 'road_damage_pipeline_results/train/processed')
    PROCESSED_TEST_DIR = os.path.join(base_dir, 'road_damage_pipeline_results/test/processed')
    PROCESSED_VALID_DIR = os.path.join(base_dir, 'road_damage_pipeline_results/valid/processed')

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 데이터 전처리 단계
    if preprocess_data:
        print("데이터 전처리를 시작합니다...")
        if not os.path.exists(PROCESSED_TRAIN_DIR):
            preprocess_images(TRAIN_DIR, PROCESSED_TRAIN_DIR)
        if not os.path.exists(PROCESSED_TEST_DIR):
            preprocess_images(TEST_DIR, PROCESSED_TEST_DIR)
        if not os.path.exists(PROCESSED_VALID_DIR):
            preprocess_images(VALID_DIR, PROCESSED_VALID_DIR)
        print("데이터 전처리가 완료되었습니다.")
    else:
        print("전처리된 데이터를 사용합니다.")

    # 분류기 초기화
    classifier = RoadDamageClassifier(
        train_dir=PROCESSED_TRAIN_DIR,
        test_dir=PROCESSED_TEST_DIR,
        valid_dir=PROCESSED_VALID_DIR,
        img_size=img_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs
    )

    # 사전 학습된 모델 로드 또는 새 모델 생성
    start_time = time.time()
    if use_pretrained and pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"사전 학습된 모델을 불러옵니다: {pretrained_model_path}")
        classifier.model = tf.keras.models.load_model(pretrained_model_path)
    else:
        print("새로운 모델을 생성합니다...")
        classifier.build_model()
    end_time = time.time()
    print(f"모델 생성 시간: {end_time - start_time:.2f}초")
    print("모델 생성이 완료되었습니다.")
    
    # 모델 학습
    history = classifier.train()
    print("모델 학습이 완료되었습니다.")
    
    # 결과 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(output_dir, f'road_damage_model_{timestamp}.keras')
    
    # 학습 결과 저장
    classifier.save_model(model_save_path)
    print(f"모델이 저장되었습니다: {model_save_path}")

    # 학습 곡선 저장
    plot_path = os.path.join(output_dir, f'training_history_{timestamp}.png')
    classifier.plot_training_history(history, save_path=plot_path)
    
    # 평가 결과
    evaluation_results = classifier.evaluate()

    print("\n모델 해석 및 분석을 시작합니다...")
    # 테스트 데이터 준비 (배치 사이즈만큼만 분석)
    _, _, test_generator = classifier.create_data_generators()
    test_batch = next(test_generator)
    test_images, test_labels = test_batch[0], test_batch[1]

    # 분석 결과 디렉토리 생성
    analysis_dir = os.path.join(output_dir, f'analysis_{timestamp}')
    os.makedirs(analysis_dir, exist_ok=True)

    return model_save_path, history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='도로 손상 분류 모델 학습')
    
    # Jupyter/Colab 환경을 위한 인자 추가
    parser.add_argument('-f', '--file', help=argparse.SUPPRESS)  # Jupyter 노트북 파일 인자 처리
    
    # 기존 인자들
    parser.add_argument('--preprocess', action='store_true',
                      help='데이터 전처리 수행')
    parser.add_argument('--use-pretrained', action='store_true',
                      help='사전 학습된 모델 사용')
    parser.add_argument('--model-path', type=str,
                      help='사전 학습된 모델 파일 경로')
    
    # 추가적인 학습 관련 인자들
    parser.add_argument('--batch-size', type=int, default=128,
                      help='배치 크기 (기본값: 32)')
    parser.add_argument('--epochs', type=int, default=300,
                      help='학습 에포크 수 (기본값: 50)')
    parser.add_argument('--img-size', type=int, default=224,
                      help='입력 이미지 크기 (기본값: 224)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                      help='학습률 (기본값: 0.001)')
    parser.add_argument('--output-dir', type=str, default='./models',
                      help='모델과 결과를 저장할 디렉토리 (기본값: ./models)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # main 함수 호출
    model_path, history = main(
        preprocess_data=args.preprocess,
        use_pretrained=args.use_pretrained,
        pretrained_model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        img_size=args.img_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    print(f"\n=== 학습 및 분석 완료 ===")
    print(f"모델 저장 경로: {model_path}")
