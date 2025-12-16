# -*- coding: utf-8 -*-
"""
功夫動作分類 - 多種分類模型比較
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)

# ML 模型
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

# XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
    print('XGBoost 已載入')
except ImportError:
    HAS_XGBOOST = False
    print('XGBoost 未安裝')

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用裝置: {device}')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 讀取資料
df = pd.read_csv('./dataset/pose_angles_summary_actual.csv')
print(f'總樣本數: {len(df)}')
print(f'\n各動作類型數量:')
print(df['Action_Type'].value_counts())

feature_columns = ['R_Elbow_Angle', 'L_Elbow_Angle', 'R_Knee_Angle', 'L_Knee_Angle', 'R_Hip_Angle', 'L_Hip_Angle']
X = df[feature_columns].values
y = df['Action_Type'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f'\n動作類別對應:')
for i, label in enumerate(label_encoder.classes_):
    print(f'  {label} -> {i}')

# 分割資料
X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=SEED, stratify=y_encoded)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=SEED, stratify=y_temp)

print(f'\n訓練集: {len(X_train)}, 驗證集: {len(X_val)}, 測試集: {len(X_test)}')

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 定義 ML 模型
ml_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'SVM (RBF)': SVC(kernel='rbf', random_state=SEED, probability=True),
    'SVM (Linear)': SVC(kernel='linear', random_state=SEED, probability=True),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=SEED),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=SEED, algorithm='SAMME')
}

if HAS_XGBOOST:
    ml_models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=5, random_state=SEED, use_label_encoder=False, eval_metric='mlogloss')

# 訓練與評估 ML 模型
results = {}
print('\n' + '='*70)
print('訓練傳統機器學習模型...')
print('='*70)

for name, model in ml_models.items():
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time

    y_pred_test = model.predict(X_test_scaled)
    y_pred_val = model.predict(X_val_scaled)

    test_acc = accuracy_score(y_test, y_pred_test)
    val_acc = accuracy_score(y_val, y_pred_val)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    results[name] = {
        'type': 'ML',
        'train_time': train_time,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred_test,
        'labels': y_test
    }

    print(f'  {name}: 準確率={test_acc*100:.2f}%, F1={test_f1*100:.2f}%, 時間={train_time:.4f}s')

# 定義 DNN
class KungfuDNN(nn.Module):
    def __init__(self, input_size=6, num_classes=4, dropout_rate=0.3):
        super(KungfuDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.network(x)

# 訓練 DNN
print('\n' + '='*70)
print('訓練深度學習模型 (DNN)...')
print('='*70)

train_tensor = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
val_tensor = TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val))
test_tensor = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))

train_loader = DataLoader(train_tensor, batch_size=16, shuffle=True)
val_loader = DataLoader(val_tensor, batch_size=16, shuffle=False)
test_loader = DataLoader(test_tensor, batch_size=16, shuffle=False)

start_time = time.time()
dnn_model = KungfuDNN(input_size=6, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(dnn_model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

best_val_acc = 0.0
patience_counter = 0
best_state = None

for epoch in range(100):
    dnn_model.train()
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = dnn_model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    dnn_model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = dnn_model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total
    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = dnn_model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= 20:
        print(f'  Early stopping at epoch {epoch+1}')
        break

if best_state:
    dnn_model.load_state_dict(best_state)

train_time = time.time() - start_time

# 評估 DNN
dnn_model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        outputs = dnn_model(features)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

y_pred_dnn = np.array(all_preds)
y_true_dnn = np.array(all_labels)

dnn_test_acc = accuracy_score(y_true_dnn, y_pred_dnn)
dnn_test_f1 = f1_score(y_true_dnn, y_pred_dnn, average='weighted')
dnn_test_precision = precision_score(y_true_dnn, y_pred_dnn, average='weighted')
dnn_test_recall = recall_score(y_true_dnn, y_pred_dnn, average='weighted')

results['DNN (PyTorch)'] = {
    'type': 'DL',
    'train_time': train_time,
    'val_acc': best_val_acc,
    'test_acc': dnn_test_acc,
    'test_f1': dnn_test_f1,
    'test_precision': dnn_test_precision,
    'test_recall': dnn_test_recall,
    'cv_mean': None,
    'cv_std': None,
    'predictions': y_pred_dnn,
    'labels': y_true_dnn
}

print(f'  DNN (PyTorch): 準確率={dnn_test_acc*100:.2f}%, F1={dnn_test_f1*100:.2f}%, 時間={train_time:.2f}s')

# 建立結果表格
results_df = pd.DataFrame({
    '模型': list(results.keys()),
    '類型': [results[k]['type'] for k in results],
    '測試準確率(%)': [round(results[k]['test_acc'] * 100, 2) for k in results],
    'F1分數(%)': [round(results[k]['test_f1'] * 100, 2) for k in results],
    '精確率(%)': [round(results[k]['test_precision'] * 100, 2) for k in results],
    '召回率(%)': [round(results[k]['test_recall'] * 100, 2) for k in results],
    '驗證準確率(%)': [round(results[k]['val_acc'] * 100, 2) for k in results],
    '訓練時間(秒)': [round(results[k]['train_time'], 4) for k in results],
    '5-fold CV': [f"{results[k]['cv_mean']*100:.2f}+/-{results[k]['cv_std']*100:.2f}" if results[k]['cv_mean'] else '-' for k in results]
})

results_df = results_df.sort_values('測試準確率(%)', ascending=False).reset_index(drop=True)

print('\n' + '='*100)
print('模型比較結果 (按準確率排序)')
print('='*100)
print(results_df.to_string(index=False))

# 儲存結果
results_df.to_csv('./model/classification_models_comparison.csv', index=False, encoding='utf-8-sig')
print('\n結果已儲存至 ./model/classification_models_comparison.csv')

# ============================================
# 繪製比較圖
# ============================================
model_names = list(results.keys())
test_accs = [results[k]['test_acc'] * 100 for k in model_names]
f1_scores_list = [results[k]['test_f1'] * 100 for k in model_names]
train_times = [results[k]['train_time'] for k in model_names]
model_types = [results[k]['type'] for k in model_names]
colors = ['#FF6B6B' if t == 'DL' else '#4ECDC4' for t in model_types]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 測試準確率
ax1 = axes[0, 0]
bars1 = ax1.barh(range(len(model_names)), test_accs, color=colors)
ax1.set_yticks(range(len(model_names)))
ax1.set_yticklabels(model_names)
ax1.set_xlabel('準確率 (%)')
ax1.set_title('測試集準確率比較', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 105)
for bar, acc in zip(bars1, test_accs):
    ax1.text(acc + 1, bar.get_y() + bar.get_height()/2, f'{acc:.1f}%', va='center', fontsize=9)

# 2. F1 分數
ax2 = axes[0, 1]
bars2 = ax2.barh(range(len(model_names)), f1_scores_list, color=colors)
ax2.set_yticks(range(len(model_names)))
ax2.set_yticklabels(model_names)
ax2.set_xlabel('F1 分數 (%)')
ax2.set_title('測試集 F1 分數比較', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 105)
for bar, f1 in zip(bars2, f1_scores_list):
    ax2.text(f1 + 1, bar.get_y() + bar.get_height()/2, f'{f1:.1f}%', va='center', fontsize=9)

# 3. 訓練時間
ax3 = axes[1, 0]
bars3 = ax3.barh(range(len(model_names)), train_times, color=colors)
ax3.set_yticks(range(len(model_names)))
ax3.set_yticklabels(model_names)
ax3.set_xlabel('訓練時間 (秒)')
ax3.set_title('訓練時間比較', fontsize=14, fontweight='bold')
for bar, t in zip(bars3, train_times):
    ax3.text(t + 0.01, bar.get_y() + bar.get_height()/2, f'{t:.3f}s', va='center', fontsize=9)

# 4. 效率散點圖
ax4 = axes[1, 1]
for name, acc, t, c in zip(model_names, test_accs, train_times, colors):
    ax4.scatter(t, acc, s=200, c=c, alpha=0.7, edgecolors='black', linewidth=1)
    ax4.annotate(name, (t, acc), textcoords='offset points', xytext=(5, 5), fontsize=8)
ax4.set_xlabel('訓練時間 (秒)')
ax4.set_ylabel('測試準確率 (%)')
ax4.set_title('效率分析: 準確率 vs 訓練時間', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4ECDC4', label='傳統 ML'), Patch(facecolor='#FF6B6B', label='深度學習')]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98), fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('./model/classification_models_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('比較圖已儲存至 ./model/classification_models_comparison.png')

# ============================================
# 混淆矩陣
# ============================================
sorted_models = sorted(results.keys(), key=lambda k: results[k]['test_acc'], reverse=True)[:6]
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
axes2 = axes2.flatten()

for idx, name in enumerate(sorted_models):
    cm = confusion_matrix(results[name]['labels'], results[name]['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=axes2[idx])
    acc = results[name]['test_acc'] * 100
    axes2[idx].set_title(f'{name}\nAcc: {acc:.1f}%', fontsize=10)
    axes2[idx].set_xlabel('預測類別')
    axes2[idx].set_ylabel('實際類別')

plt.tight_layout()
plt.savefig('./model/confusion_matrices_all_models.png', dpi=150, bbox_inches='tight')
plt.close()
print('混淆矩陣已儲存至 ./model/confusion_matrices_all_models.png')

# ============================================
# 各類別表現
# ============================================
class_metrics = {}
for name, data in results.items():
    report = classification_report(
        data['labels'], data['predictions'],
        target_names=label_encoder.classes_,
        output_dict=True
    )
    for cls in label_encoder.classes_:
        if cls not in class_metrics:
            class_metrics[cls] = {}
        class_metrics[cls][name] = report[cls]['f1-score']

fig3, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(model_names))
width = 0.2
multiplier = 0

action_names = {'act1': '握拳式', 'act2': '出拳式', 'act3': '踢腿式', 'act4': '提膝式'}
colors_class = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i, (cls, metrics) in enumerate(class_metrics.items()):
    offset = width * multiplier
    f1_values = [metrics[name] * 100 for name in model_names]
    ax.bar(x + offset, f1_values, width, label=f'{cls} ({action_names[cls]})', color=colors_class[i])
    multiplier += 1

ax.set_ylabel('F1 分數 (%)')
ax.set_title('各類別在不同模型上的 F1 分數比較', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend(loc='lower right')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('./model/class_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('各類別表現比較圖已儲存至 ./model/class_performance_comparison.png')

# ============================================
# 雷達圖
# ============================================
top_models = sorted(results.keys(), key=lambda k: results[k]['test_acc'], reverse=True)[:5]
categories = ['準確率', 'F1分數', '精確率', '召回率']
num_vars = len(categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig4, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
colors_radar = plt.cm.Set2(np.linspace(0, 1, len(top_models)))

for i, name in enumerate(top_models):
    values = [
        results[name]['test_acc'] * 100,
        results[name]['test_f1'] * 100,
        results[name]['test_precision'] * 100,
        results[name]['test_recall'] * 100
    ]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors_radar[i])
    ax.fill(angles, values, alpha=0.1, color=colors_radar[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 105)
ax.set_title('Top 5 模型效能雷達圖', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('./model/model_radar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print('雷達圖已儲存至 ./model/model_radar_chart.png')

# ============================================
# 結論
# ============================================
print('\n' + '='*70)
print('結論與建議')
print('='*70)

best_acc_model = max(results.keys(), key=lambda k: results[k]['test_acc'])
best_f1_model = max(results.keys(), key=lambda k: results[k]['test_f1'])
fastest_model = min(results.keys(), key=lambda k: results[k]['train_time'])

ml_models_results = {k: v for k, v in results.items() if v['type'] == 'ML'}
dl_models_results = {k: v for k, v in results.items() if v['type'] == 'DL'}

best_ml_model = max(ml_models_results.keys(), key=lambda k: ml_models_results[k]['test_acc'])
best_dl_model = max(dl_models_results.keys(), key=lambda k: dl_models_results[k]['test_acc']) if dl_models_results else None

print(f'\n🏆 最高準確率模型: {best_acc_model}')
print(f'   測試準確率: {results[best_acc_model]["test_acc"]*100:.2f}%')
print(f'   F1 分數: {results[best_acc_model]["test_f1"]*100:.2f}%')

print(f'\n🎯 最高 F1 分數模型: {best_f1_model}')
print(f'   F1 分數: {results[best_f1_model]["test_f1"]*100:.2f}%')

print(f'\n⚡ 最快訓練模型: {fastest_model}')
print(f'   訓練時間: {results[fastest_model]["train_time"]:.4f} 秒')
print(f'   準確率: {results[fastest_model]["test_acc"]*100:.2f}%')

print(f'\n📊 最佳傳統 ML 模型: {best_ml_model}')
print(f'   測試準確率: {results[best_ml_model]["test_acc"]*100:.2f}%')

if best_dl_model:
    print(f'\n🧠 最佳深度學習模型: {best_dl_model}')
    print(f'   測試準確率: {results[best_dl_model]["test_acc"]*100:.2f}%')

print('\n' + '-'*70)
print('建議:')
print('-'*70)

if best_dl_model and results[best_dl_model]['test_acc'] > results[best_ml_model]['test_acc']:
    print(f'\n✅ 深度學習模型 ({best_dl_model}) 表現最佳')
    print(f'   但訓練時間較長 ({results[best_dl_model]["train_time"]:.2f}s)')
else:
    print(f'\n✅ 傳統 ML 模型 ({best_ml_model}) 表現最佳')
    print(f'   且訓練速度快 ({results[best_ml_model]["train_time"]:.4f}s)')

print(f'\n📌 對於此功夫動作分類任務:')
print(f'   - 若追求最高準確率: 使用 {best_acc_model}')
print(f'   - 若需要快速訓練: 使用 {fastest_model}')
print(f'   - 若需平衡效能與速度: 使用 {best_ml_model}')

print('\n' + '='*70)
print('所有比較完成！')
print('='*70)
