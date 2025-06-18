# CQT実験の技術スタック詳細解説

## 概要

CQT（複素量子軌跡）実験は以下の3つの主要技術で構成されています：

1. **NumPy**: 数値計算とデータ処理
2. **matplotlib**: データ可視化
3. **独自CQTアルゴリズム**: 量子測定を複素数に変換

## 1. NumPy の役割

### 基本的な数値計算
```python
# 量子状態ベクトルの正規化
state_vector = state_vector / np.linalg.norm(state_vector)

# 測定確率の計算
prob_0 = np.abs(state_vector[0])**2
prob_1 = np.abs(state_vector[1])**2

# ランダム測定結果の生成
outcomes = np.random.choice([0, 1], size=n_measurements, p=[prob_0, prob_1])
```

### 複素数演算
```python
# 複素軌跡の作成
trajectory_array = np.array(self.trajectory)
real_parts = trajectory_array.real
imag_parts = trajectory_array.imag

# 位相と振幅の計算
magnitudes = np.abs(trajectory_array)
phases = np.angle(trajectory_array)
```

### 統計解析
```python
# 軌跡の統計的特性
mean_complex = np.mean(trajectory_array)
std_complex = np.std(trajectory_array)
correlation = np.correlate(real_parts, real_parts, mode='same')
```

## 2. matplotlib の役割

### 複素平面での軌跡可視化
```python
# カラーグラデーションでの軌跡プロット
colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
for i in range(len(trajectory) - 1):
    ax.plot(real_parts[i:i+2], imag_parts[i:i+2], 
           color=colors[i], alpha=0.7, linewidth=2)
```

### 時系列データの表示
```python
# 大きさと位相の時間変化
ax.plot(magnitudes, label='Magnitude', color='blue')
ax_twin = ax.twinx()
ax_twin.plot(phases, label='Phase', color='orange')
```

### アニメーション機能
```python
# リアルタイム軌跡進化の可視化
from matplotlib.animation import FuncAnimation
anim = FuncAnimation(fig, animate, frames=n_frames, interval=50)
```

## 3. 独自CQTアルゴリズム

### 核心的な複素数変換アルゴリズム

#### 方向性（実部）の計算
```python
def _compute_direction(self, outcome: int, index: int) -> float:
    # 基本方向：0→-1, 1→+1
    base_direction = 2 * outcome - 1
    
    # 時間的変調（周期100測定）
    temporal_factor = np.cos(2 * np.pi * index / 100)
    
    # 履歴効果（過去10測定の影響）
    if len(self.trajectory) > 0:
        recent_history = self.trajectory[-10:]
        history_effect = np.mean([z.real for z in recent_history]) * 0.3
    
    return base_direction + 0.2 * temporal_factor + history_effect
```

#### 不確実性（虚部）の計算
```python
def _compute_uncertainty(self, outcome: int, state_vector=None) -> float:
    if state_vector is not None:
        # 量子状態から不確実性を計算
        probabilities = np.abs(state_vector) ** 2
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        uncertainty = entropy / np.log(self.system_dim)
    else:
        # 測定統計から不確実性を推定
        recent_outcomes = [m.outcome for m in self.measurements[-10:]]
        p0 = recent_outcomes.count(0) / len(recent_outcomes)
        p1 = recent_outcomes.count(1) / len(recent_outcomes)
        # バイナリエントロピー
        uncertainty = -(p0*np.log(p0+1e-10) + p1*np.log(p1+1e-10)) / np.log(2)
    
    return uncertainty
```

## なぜQiskitを使わないのか？

### 設計思想の違い

**Qiskit**:
- 量子回路の構築と実行に特化
- 量子ゲート操作が中心
- 測定結果は従来の0/1バイナリ

**CQTアプローチ**:
- 測定結果の「質」に着目
- 複素数による情報保持
- 軌跡解析による新しい洞察

### 技術的利点

1. **軽量性**
   - NumPy/matplotlibのみで動作
   - 依存関係が少ない
   - 高速な計算

2. **柔軟性**
   - アルゴリズムを自由にカスタマイズ
   - リアルタイム解析が容易
   - 実験的機能の迅速な実装

3. **移植性**
   - 任意の量子プラットフォームと統合可能
   - IBM Quantum、Google Cirq、IonQなど
   - 古典シミュレーターでの高速検証

## CQTアルゴリズムの革新性

### 1. 情報の保持
従来：測定 → {0, 1} → 情報損失
CQT：測定 → z = a + bi → 情報保持

### 2. パターン認識
```python
# 量子状態の「指紋」検出
def detect_state_signature(trajectory):
    if np.mean(np.abs(trajectory.real)) < 0.1:  # |+⟩状態
        return "superposition"
    elif np.mean(trajectory.real) > 0.5:       # |1⟩状態
        return "excited"
    elif np.mean(trajectory.real) < -0.5:      # |0⟩状態
        return "ground"
```

### 3. エラー検出
```python
# 軌跡パターンからエラー検出
def detect_quantum_error(trajectory_window):
    distances = [abs(z) for z in trajectory_window[-5:]]
    if all(d < 0.5 for d in distances):
        return "DECOHERENCE_DETECTED"
    
    phases = [np.angle(z) for z in trajectory_window]
    if np.std(np.diff(phases)) > np.pi/2:
        return "PHASE_FLIP_DETECTED"
```

## 実装上の工夫

### 数値安定性
```python
# ゼロ除算回避
entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))

# 正規化
uncertainty = entropy / np.log(self.system_dim)
```

### メモリ効率
```python
# 軌跡データの効率的な格納
@dataclass
class MeasurementRecord:
    measurement_index: int
    outcome: int
    complex_value: complex
    timestamp: float = 0.0
```

### リアルタイム性能
```python
# 履歴効果の高速計算（最新10点のみ使用）
recent_history = self.trajectory[-10:]
history_effect = np.mean([z.real for z in recent_history]) * 0.3
```

## まとめ

この技術スタックは：
- **軽量で高速**な量子測定解析を実現
- **新しい物理洞察**を提供する独自アルゴリズム
- **実用的な応用**（エラー検出、状態識別）が可能
- **将来の拡張**（多量子ビット、実機統合）に対応

CQT理論の実装として、従来の量子計算フレームワークとは異なる革新的なアプローチを提供しています。