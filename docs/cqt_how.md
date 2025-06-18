# CQT理論における複素数値の決定方法

**文書作成日**: 2025年12月19日  
**理論開発者**: 大谷 圭亮  
**文書バージョン**: 1.0

## 概要

CQT理論（Complex Quantum Ternary Theory / 複素量子三値理論）では、量子測定結果を複素数 `z = a + bi` として表現します。本文書では、プロジェクトナレッジから抽出した複素数値の具体的な決定方法を体系的にまとめます。

## 1. 基本的な複素数表現

### 1.1 複素数の構成要素

```
z = a + bi
```

- **a (実部)**: 判定の方向性 (-1 ～ +1)
- **b (虚部)**: 不確実性の度合い (0 ～ 1)

### 1.2 物理的意味

- **実部が正**: |1⟩状態に近い
- **実部が負**: |0⟩状態に近い
- **実部がゼロ**: 完全に不確定
- **虚部が大きい**: 測定の不確実性が高い
- **虚部が小さい**: 測定が確定的

## 2. 量子測定における具体的な変換方法

### 2.1 基本的な確率ベース変換

```python
def quantum_to_complex_ternary(measurement_result):
    """
    量子測定結果を複素3値に変換
    """
    # 測定で得られた確率
    prob_0 = measurement_result['prob_0']  # |0⟩の確率
    prob_1 = measurement_result['prob_1']  # |1⟩の確率
    
    # 方法1: 確率差を実部、エントロピーを虚部
    real_part = prob_1 - prob_0  # -1 to 1
    entropy = -prob_0*np.log2(prob_0+1e-10) - prob_1*np.log2(prob_1+1e-10)
    imag_part = entropy  # 0 to 1
    
    return real_part + imag_part * 1j
```

### 2.2 位相情報を使う方法

```python
def quantum_phase_to_complex(quantum_state):
    """
    量子位相を直接利用
    """
    # 量子状態の複素振幅
    alpha = quantum_state.amplitude_0  # 複素数
    beta = quantum_state.amplitude_1   # 複素数
    
    # 相対位相を利用
    relative_phase = np.angle(beta / alpha) if alpha != 0 else np.pi/2
    magnitude = abs(beta)**2 - abs(alpha)**2
    
    # 実部：確率の差
    # 虚部：位相の不確定性
    real_part = magnitude
    imag_part = np.sin(relative_phase)
    
    return real_part + imag_part * 1j
```

### 2.3 完全な量子測定の変換（プロジェクトで使用）

```python
def complete_quantum_to_complex(qubit_measurement):
    """
    量子測定の完全な複素3値変換
    """
    # 生の測定データ
    counts = qubit_measurement['counts']  # {'0': 512, '1': 488}
    total = sum(counts.values())
    
    # 確率計算
    p0 = counts.get('0', 0) / total
    p1 = counts.get('1', 0) / total
    
    # 実部：期待値（0または1への偏り）
    expectation = p1  # 0 to 1
    real_part = 2 * expectation - 1  # -1 to 1
    
    # 虚部：測定の不確実性
    # 方法1: エントロピー
    entropy = -p0*np.log2(p0+1e-10) - p1*np.log2(p1+1e-10)
    
    # 方法2: 分散
    variance = p0 * p1  # 最大0.25 when p0=p1=0.5
    
    # 方法3: 統計的不確実性
    statistical_uncertainty = np.sqrt(p0 * p1 / total)
    
    # 組み合わせ
    imag_part = 0.5 * entropy + 0.3 * (4 * variance) + 0.2 * statistical_uncertainty
    
    return real_part + imag_part * 1j
```

## 3. 応用分野での係数定義

### 3.1 医学診断での実装

```python
def medical_to_complex_ternary(diagnosis):
    """
    医学診断を複素3値に変換
    """
    # 実部：診断の確信度（-1: 陰性確定, +1: 陽性確定）
    confidence = diagnosis['confidence']
    diagnosis_direction = 1 if diagnosis['result'] == 'positive' else -1
    real_part = confidence * diagnosis_direction
    
    # 虚部：追加検査の必要性（不確実性の指標）
    test_reliability = diagnosis['test_reliability']
    symptom_ambiguity = diagnosis['symptom_ambiguity']
    imag_part = (1 - test_reliability) * symptom_ambiguity
    
    return real_part + imag_part * 1j
```

### 3.2 金融リスク評価での実装

```python
def financial_to_complex_ternary(investment):
    """
    金融判断を複素3値に変換
    """
    # 実部：期待リターン（正規化）
    expected_return = investment['expected_return']
    real_part = np.tanh(expected_return)  # -1 to 1
    
    # 虚部：ボラティリティ（不確実性）
    volatility = investment['volatility']
    imag_part = volatility / (1 + volatility)  # 0 to 1
    
    return real_part + imag_part * 1j
```

## 4. 統一的な変換原理

### 4.1 基本原則

1. **実部の意味（判定の方向）**
   - `-1`：完全に否定的（0, 陰性, 売り）
   - `0`：中立（判定不能）
   - `+1`：完全に肯定的（1, 陽性, 買い）

2. **虚部の意味（不確実性）**
   - `0`：完全に確実
   - `0.5`：中程度の不確実性
   - `1`：最大の不確実性

### 4.2 統一インターフェース

```python
class UniversalComplexTernaryConverter:
    """
    あらゆる測定を複素3値に変換する統一インターフェース
    """
    
    def convert(self, measurement, domain):
        """
        統一的な変換メソッド
        """
        # 基本原則：
        # 実部 = 判定の方向性（-1 to +1）
        # 虚部 = 判定の不確実性（0 to 1）
        
        # ステップ1: 判定スコアを計算
        decision_score = self._calculate_decision_score(measurement, domain)
        
        # ステップ2: 不確実性を計算
        uncertainty = self._calculate_uncertainty(measurement, domain)
        
        # ステップ3: 複素数として結合
        z = decision_score + uncertainty * 1j
        
        return z
```

## 5. CQT実験プロジェクトでの実装

### 5.1 測定軌跡追跡での使用例

```python
class CQTMeasurementTracker:
    def measure_to_complex(self, counts, shot_index, total_shots=1):
        """
        測定結果を複素数に変換
        """
        # カウントの正規化
        p0 = counts.get('0', 0) / total_shots
        p1 = counts.get('1', 0) / total_shots
        
        # 方向性：|1⟩寄りなら正、|0⟩寄りなら負
        a = p1 - p0
        
        # 不確実性：50/50に近いほど大きい
        if p0 > 0 and p1 > 0:
            entropy = -p0 * np.log2(p0) - p1 * np.log2(p1)
            b = entropy
        else:
            b = 0.0
        
        return complex(a, b)
```

### 5.2 複素数の解釈

```python
def interpret_complex_ternary(z):
    """
    複素3値の解釈
    """
    magnitude = abs(z)  # 情報の総量
    phase = np.angle(z)  # 判定の性質
    
    # 判定
    if abs(z.real) > 0.8 and abs(z.imag) < 0.2:
        return "確定的判定"
    elif abs(z.imag) > 0.7:
        return "高不確実性につき保留"
    else:
        return "傾向はあるが要検証"
```

## 6. 重要な設計原則

### 6.1 値の範囲と意味

| 成分 | 範囲 | 意味 |
|------|------|------|
| 実部 | [-1, 1] | -1 (完全に0) ～ 0 (不確定) ～ 1 (完全に1) |
| 虚部 | [0, 1] | 0 (確実) ～ 0.5 (中程度の不確実性) ～ 1 (最大不確実性) |

### 6.2 係数の選択指針

1. **エントロピー係数**: 0.5（基本的な不確実性の指標）
2. **分散係数**: 0.3（統計的ばらつきの考慮）
3. **統計誤差係数**: 0.2（有限サンプルの影響）

これらの係数は実験的に調整可能で、応用分野により最適化されます。

## 7. まとめ

CQT理論における複素数値の決定は、以下の原則に基づきます：

1. **統一的な表現**: 実部に判定方向、虚部に不確実性を割り当てる
2. **物理的意味の保持**: 量子測定の確率的性質を正確に反映
3. **応用の柔軟性**: 各分野の特性に応じた係数調整が可能
4. **数学的一貫性**: あらゆる測定を同一の枠組みで扱える

この方法により、量子測定の豊かな情報を失うことなく、直感的で扱いやすい形式で表現することが可能になります。

---

**参考資料**:
- プロジェクトナレッジ文書番号: 16, 19, 25
- CQT理論実験プロジェクト実装コード