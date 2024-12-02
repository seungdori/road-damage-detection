import torch
import torch.nn as nn
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import json
from datetime import datetime
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import os
import shutil
from matplotlib import rc
import matplotlib.font_manager as fm
# 시스템에 설치된 모든 글꼴 경로 출력

# 한글 글꼴 설정
rc('font', family='AppleGothic') 			
plt.rcParams['axes.unicode_minus'] = False  


class RoadDamageDataset(Dataset):
    """도로 파손 데이터셋 클래스"""
    
    def __init__(self, image_dir, label_dir, transform=None, phase='train'):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.phase = phase
    
        
        # 이미지 파일 리스트 생성
        self.image_files = list(self.image_dir.glob('*.jpg'))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        
        # 이미지 로드
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 라벨 로드
        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([x_center, y_center, width, height])
                    labels.append(int(class_id))
        
        # numpy 배열로 변환
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # 데이터 증강 적용
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        return {
            'image': image,
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.LongTensor(labels)
        }


def get_transform(phase):
    """데이터 증강 파이프라인 정의"""
    if phase == 'train':
        return A.Compose([
            A.RandomResizedCrop(416, 416, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomRotate90(p=0.2),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(p=1.0),
            ], p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.Normalize(),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(416, 416),
            A.Normalize(),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))


def collate_fn(batch):
    """배치 데이터 처리를 위한 콜레이트 함수"""
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'boxes': [item['boxes'] for item in batch],
        'labels': [item['labels'] for item in batch]
    }

class RoadDamageProcessor:
    """도로 파손 데이터 분석 및 전처리를 위한 통합 클래스"""
    
    def __init__(self, image_dir, label_dir, output_dir='processed_data'):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        
        # 결과 저장을 위한 디렉토리 구조 생성
        self.dirs = {
            'analysis': self.output_dir / 'analysis',
            'plots': self.output_dir / 'analysis/plots',
            'processed': self.output_dir / 'processed',
            'samples': self.output_dir / 'samples'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.class_names = {
            0: 'Longitudinal Cracks (D00)',
            1: 'Transverse Cracks (D10)',
            2: 'Alligator Cracks (D20)',
            3: 'Potholes (D40)',
            4: 'Damaged Crosswalk (D43)',
            5: 'Damaged Paint (D44)',
            6: 'Manhole Cover (D50)'
        }
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        

    def save_sample_images(self, num_samples=5):
        """각 클래스별 샘플 이미지 저장"""
        class_samples = defaultdict(list)
        
        # 각 클래스별로 이미지 수집
        for image_file in self.image_dir.glob('*.jpg'):
            label_file = self.label_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                image = cv2.imread(str(image_file))
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id = int(float(line.strip().split()[0]))
                        class_samples[class_id].append((image_file, line))
        
        # 각 클래스별로 샘플 이미지 저장
        for class_id in self.class_names.keys():
            samples = class_samples.get(class_id, [])
            if not samples:
                continue
                
            # 샘플 선택
            selected_samples = samples[:min(num_samples, len(samples))]
            
            # 클래스별 디렉토리 생성
            class_dir = self.samples_dir / f"class_{class_id}_{self.class_names[class_id].replace(' ', '_')}"
            class_dir.mkdir(exist_ok=True)
            
            for idx, (image_file, label_line) in enumerate(selected_samples):
                image = cv2.imread(str(image_file))
                
                # 바운딩 박스 그리기
                h, w = image.shape[:2]
                class_id, x_center, y_center, width, height = map(float, label_line.strip().split())
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # 바운딩 박스와 레이블 추가
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, self.class_names[int(class_id)], (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 이미지 저장
                cv2.imwrite(str(class_dir / f"sample_{idx}.jpg"), image)
                
                # 크롭된 이미지도 저장
                cropped = image[y1:y2, x1:x2]
                cv2.imwrite(str(class_dir / f"sample_{idx}_cropped.jpg"), cropped)

    def _save_analysis_results(self, stats):
            """분석 결과를 JSON 파일로 저장합니다."""
            # 통계 데이터 정리
            analysis_results = {
                'dataset_overview': {
                    'total_images': len(stats['image_sizes']),
                    'total_objects': sum(stats['class_distribution'].values()),
                    'avg_objects_per_image': np.mean(stats['boxes_per_image']),
                    'creation_date': self.timestamp
                },
                'class_distribution': {
                    self.class_names[class_id]: count 
                    for class_id, count in stats['class_distribution'].items()
                },
                'size_statistics': {
                    'avg_box_size': float(np.mean(stats['box_sizes'])),
                    'avg_aspect_ratio': float(np.mean(stats['aspect_ratios'])),
                    'image_size_distribution': {
                        'heights': [int(h) for h, w in stats['image_sizes']],
                        'widths': [int(w) for h, w in stats['image_sizes']]
                    }
                }
            }

            # JSON 파일로 저장
            output_file = self.dirs['analysis'] / f'analysis_results_{self.timestamp}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=4, ensure_ascii=False)
                
            print(f"분석 결과가 저장되었습니다: {output_file}")

    def _visualize_analysis(self, stats):
        """분석 결과를 시각화하여 그래프로 저장합니다."""
        plt.style.use('default')
        
        # 1. 클래스별 분포 그래프
        plt.figure(figsize=(15, 8))
        class_names = [self.class_names[i] for i in sorted(stats['class_distribution'].keys())]
        class_counts = [stats['class_distribution'][i] for i in sorted(stats['class_distribution'].keys())]
        
        plt.bar(class_names, class_counts)
        plt.title('클래스별 객체 분포')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('객체 수')
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / f'class_distribution_{self.timestamp}.png')
        plt.close()
        
        # 2. 이미지당 객체 수 분포
        plt.figure(figsize=(10, 6))
        plt.hist(stats['boxes_per_image'], bins=30, edgecolor='black')
        plt.title('이미지당 객체 수 분포')
        plt.xlabel('객체 수')
        plt.ylabel('이미지 수')
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / f'objects_per_image_{self.timestamp}.png')
        plt.close()
        
        # 3. 바운딩 박스 크기 분포
        plt.figure(figsize=(10, 6))
        plt.hist(stats['box_sizes'], bins=50, edgecolor='black')
        plt.title('바운딩 박스 크기 분포')
        plt.xlabel('박스 크기 (정규화된 면적)')
        plt.ylabel('빈도')
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / f'box_sizes_{self.timestamp}.png')
        plt.close()
        
        # 4. 이미지 크기 산점도
        plt.figure(figsize=(10, 6))
        heights, widths = zip(*stats['image_sizes'])
        plt.scatter(widths, heights, alpha=0.5)
        plt.title('이미지 크기 분포')
        plt.xlabel('너비 (픽셀)')
        plt.ylabel('높이 (픽셀)')
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / f'image_sizes_{self.timestamp}.png')
        plt.close()
        
        print(f"시각화 결과가 {self.dirs['plots']} 디렉토리에 저장되었습니다.")

    def _save_sample_images(self, samples_per_class=5):
        """각 클래스별 샘플 이미지를 저장합니다."""
        # 클래스별 샘플 수집
        class_samples = defaultdict(list)
        
        for image_file in self.image_dir.glob('*.jpg'):
            label_file = self.label_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                image = cv2.imread(str(image_file))
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id = int(float(line.strip().split()[0]))
                        if len(class_samples[class_id]) < samples_per_class:
                            class_samples[class_id].append((image_file, line))
        
        # 각 클래스별로 샘플 이미지 저장
        for class_id in self.class_names.keys():
            samples = class_samples.get(class_id, [])
            if not samples:
                continue
            
            # 클래스별 디렉토리 생성
            class_dir = self.dirs['samples'] / f"class_{class_id}_{self.class_names[class_id].replace(' ', '_')}"
            class_dir.mkdir(exist_ok=True)
            
            # 샘플 이미지 저장
            for idx, (image_file, label_line) in enumerate(samples):
                image = cv2.imread(str(image_file))
                h, w = image.shape[:2]
                
                # 바운딩 박스 정보 추출
                class_id, x_center, y_center, width, height = map(float, label_line.strip().split())
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # 바운딩 박스 그리기
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, self.class_names[int(class_id)], 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 전체 이미지와 크롭된 이미지 저장
                cv2.imwrite(str(class_dir / f"sample_{idx}.jpg"), image)
                cropped = image[y1:y2, x1:x2]
                cv2.imwrite(str(class_dir / f"sample_{idx}_cropped.jpg"), cropped)
        
        print(f"샘플 이미지가 {self.dirs['samples']} 디렉토리에 저장되었습니다.")

    def analyze_and_save(self):
        """데이터셋 분석 수행 및 결과 저장"""
        print("데이터셋 분석 중...")
        stats = self._analyze_dataset()
        self._save_analysis_results(stats)
        self._visualize_analysis(stats)
        self._save_sample_images()
        return stats
    
    def create_dataset(self):
        """전처리된 데이터셋 생성"""
        print("데이터셋 생성 중...")
        train_loader, val_loader = self._create_dataloaders()
        
        # 데이터 로더 정보 저장
        dataset_info = {
            'train_size': len(train_loader.dataset),
            'val_size': len(val_loader.dataset),
            'image_size': (416, 416),
            'batch_size': train_loader.batch_size,
            'num_classes': len(self.class_names),
            'creation_date': self.timestamp
        }
        
        # 데이터셋 정보 저장
        with open(self.dirs['processed'] / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        return train_loader, val_loader
    
    def _analyze_dataset(self):
        """데이터셋 분석 수행"""
        stats = {
            'class_distribution': defaultdict(int),
            'image_sizes': [],
            'boxes_per_image': [],
            'aspect_ratios': [],
            'box_sizes': []
        }
        
        for image_file in self.image_dir.glob('*.jpg'):
            image = cv2.imread(str(image_file))
            stats['image_sizes'].append(image.shape[:2])
            
            label_file = self.label_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    boxes = f.readlines()
                    stats['boxes_per_image'].append(len(boxes))
                    
                    for box in boxes:
                        values = map(float, box.strip().split())
                        class_id = int(next(values))
                        x_center, y_center, width, height = tuple(values)
                        
                        stats['class_distribution'][class_id] += 1
                        stats['aspect_ratios'].append(width/height)
                        stats['box_sizes'].append(width * height)
        
        return stats
    
    def _save_processed_data(self, image, boxes, labels, idx):
        """전처리된 데이터를 클래스별로 저장하는 메서드"""
        for class_id in range(len(self.class_names)):
            # 각 클래스별 디렉토리 생성
            class_dir = self.dirs['processed'] / f'class_{class_id}_{self.class_names[class_id].replace(" ", "_")}'
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # 현재 클래스의 객체만 필터링
            class_indices = [i for i, label in enumerate(labels) if label == class_id]
            if not class_indices:
                continue
                
            # 해당 클래스의 바운딩 박스만 추출하여 저장
            for box_idx, box_i in enumerate(class_indices):
                box = boxes[box_i]
                
                # 이미지에서 바운딩 박스 부분 추출
                if torch.is_tensor(image):
                    image_np = image.numpy().transpose(1, 2, 0)
                    # 정규화 해제
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image_np = std * image_np + mean
                    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                else:
                    image_np = image
                    
                h, w = image_np.shape[:2]
                x1 = int(max((box[0] - box[2]/2) * w, 0))
                y1 = int(max((box[1] - box[3]/2) * h, 0))
                x2 = int(min((box[0] + box[2]/2) * w, w))
                y2 = int(min((box[1] + box[3]/2) * h, h))
                
                # 바운딩 박스 영역 추출
                cropped = image_np[y1:y2, x1:x2]
                
                # 크기 조정
                cropped = cv2.resize(cropped, (416, 416))
                
                # 이미지 저장
                save_path = class_dir / f"{idx:06d}_{box_idx}.jpg"
                cv2.imwrite(str(save_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    def process_and_save_dataset(self, batch_size=16):
        """데이터셋 처리 및 저장"""
        # 기본 전처리 변환 정의
        transform = A.Compose([
            A.Resize(416, 416),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.8),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(p=0.2),
            ], p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

        # 데이터셋 생성
        dataset = RoadDamageDataset(
            image_dir=self.image_dir,
            label_dir=self.label_dir,
            transform=transform
        )
        
        # 데이터 처리 및 저장
        print("전처리된 데이터 저장 중...")
        class_counts = defaultdict(int)
        
        for idx in range(len(dataset)):
            data = dataset[idx]
            self._save_processed_data(data['image'], data['boxes'], data['labels'], idx)
            
            # 클래스별 카운트 업데이트
            for label in data['labels']:
                class_counts[label.item()] += 1
                
            if idx % 100 == 0:
                print(f"처리 진행률: {idx}/{len(dataset)}")
        
        # 데이터셋 정보 저장
        dataset_info = {
            'total_images': len(dataset),
            'class_distribution': {
                self.class_names[class_id]: count 
                for class_id, count in class_counts.items()
            },
            'image_size': (416, 416),
            'creation_date': self.timestamp
        }
        
        with open(self.dirs['processed'] / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        print(f"전처리된 데이터가 {self.dirs['processed']} 디렉토리에 저장되었습니다.")

    def _create_dataloaders(self, batch_size=16):
        """데이터 로더를 생성하고 전처리된 데이터를 저장"""
        # 기본 전처리 변환 정의
        transform = A.Compose([
            A.Resize(416, 416),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.8),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(p=0.2),
            ], p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

        # 데이터셋 생성
        dataset = RoadDamageDataset(
            image_dir=self.image_dir,
            label_dir=self.label_dir,
            transform=transform
        )
        
        # 데이터셋 분할
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # 전처리된 데이터 저장
        print("전처리된 데이터 저장 중...")
        
        # 학습 데이터 저장
        train_files = []
        for idx in range(len(train_dataset)):
            data = train_dataset[idx]
            image_path, label_path = self._save_processed_data(
                data['image'], data['boxes'], data['labels'], 
                idx, split='train'
            )
            train_files.append({'image': image_path, 'label': label_path})
            if idx % 100 == 0:
                print(f"학습 데이터 저장 진행률: {idx}/{len(train_dataset)}")
        
        # 검증 데이터 저장
        val_files = []
        for idx in range(len(val_dataset)):
            data = val_dataset[idx]
            image_path, label_path = self._save_processed_data(
                data['image'], data['boxes'], data['labels'], 
                idx, split='val'
            )
            val_files.append({'image': image_path, 'label': label_path})
            if idx % 100 == 0:
                print(f"검증 데이터 저장 진행률: {idx}/{len(val_dataset)}")
        
        # 데이터셋 정보 저장
        dataset_info = {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'image_size': (416, 416),
            'train_files': train_files,
            'val_files': val_files,
            'creation_date': self.timestamp
        }
        
        with open(self.dirs['processed'] / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        print(f"전처리된 데이터가 {self.dirs['processed']} 디렉토리에 저장되었습니다.")
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
def main(image_dir, label_dir, output_dir):
    # 프로세서 인스턴스 생성
    image_dir = '/Users/seungdori/Hancom/road-damage-detection/data/valid/images'
    label_dir = '/Users/seungdori/Hancom/road-damage-detection/data/valid/labels'
    output_dir = 'road_damage_pipeline_results'
    processor = RoadDamageProcessor(
        image_dir,
        label_dir,
        output_dir
    )

    # 1단계: 데이터셋 분석
    print("1단계: 데이터셋 분석 중...")
    stats = processor.analyze_and_save()
    
    # 2단계: 데이터 전처리 및 저장
    print("2단계: 데이터 전처리 및 저장 중...")
    processor.process_and_save_dataset()
    
    print("\n처리가 완료되었습니다!")
    print(f"결과물은  {output_dir} 디렉토리에서 확인할 수 있습니다.")
    print("\n저장된 결과물:")
    print(f"1. 데이터 분석 결과: {output_dir}/analysis/")
    print(f"2. 전처리된 이미지: {output_dir}/processed/")
    print(f"3. 클래스별 샘플 이미지: road_damage_pipeline_results/samples/")

if __name__ == "__main__":
    main()