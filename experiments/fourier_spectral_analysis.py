#!/usr/bin/env python3
"""
実データのフーリエ解析とスペクトル特性解析
Fourier Analysis and Spectral Characteristics of Real Data

実際のBell状態データとIBM Quantum Volumeデータの複素軌跡に対して、
フーリエ変換とスペクトル解析を実行し、周波数領域での特性を詳細に調べます。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# scipyのインポート
try:
    from scipy import signal as sp_signal
    from scipy.fft import fft, fftfreq, fftshift
    scipy_available = True
    print("✓ Scipy利用可能")
except ImportError:
    scipy_available = False
    print("⚠ Scipy未インストール - 基本フーリエ解析のみ実行")

# プロジェクトパスの設定
sys.path.append('src')

# カスタムモジュールのインポート
try:
    from src.cqt_tracker_v3 import OptimizedCQTTracker
    print("✓ CQTモジュール読み込み成功")
except ImportError as e:
    print(f"⚠ モジュールインポートエラー: {e}")

# プロット設定
plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 12

def load_trajectories_for_fourier():
    """フーリエ解析用の軌跡データを読み込み"""
    trajectories = {}
    
    # Bell状態データ
    bell_data_path = 'data_collection/collected_data/bell_states/bell_measurement_data.csv'
    
    if os.path.exists(bell_data_path):
        bell_data = pd.read_csv(bell_data_path)
        print(f"Bell状態データ読み込み: {len(bell_data)} 状態")
        
        for idx, row in bell_data.iterrows():
            state = row['state']
            counts_str = row['counts']
            
            try:
                import ast
                counts_str = counts_str.replace("np.str_('", "'").replace("np.int64(", "").replace("')", "'").replace(")", "")
                counts = ast.literal_eval(counts_str)
                
                tracker = OptimizedCQTTracker(system_dim=2)
                
                for outcome_str, count in counts.items():
                    sample_count = min(count // 8, 256)  # フーリエ解析に適した長さ
                    for _ in range(sample_count):
                        outcome = int(outcome_str[0])
                        tracker.add_measurement(outcome)
                
                if tracker.trajectory and len(tracker.trajectory) > 128:
                    trajectories[f'bell_{state}'] = np.array(tracker.trajectory)
                    print(f"  {state}: {len(tracker.trajectory)}点")
            
            except Exception as e:
                print(f"  {state} エラー: {e}")
    
    # IBM Quantum Volumeデータ
    qv_data_path = 'data_collection/downloaded_quantum_data/IBM_Quantum_Volume/qiskit-experiments/qiskit-experiments-main/test/library/quantum_volume'
    
    qv_files = {
        'qv_moderate': 'qv_data_moderate_noise_100_trials.json',
        'qv_high_noise': 'qv_data_high_noise.json'
    }
    
    for label, filename in qv_files.items():
        filepath = os.path.join(qv_data_path, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                print(f"{label}: {len(data)} 試行読み込み")
                
                tracker = OptimizedCQTTracker(system_dim=4)
                
                for trial_idx in range(min(4, len(data))):
                    trial = data[trial_idx]
                    
                    if 'counts' in trial:
                        counts = trial['counts']
                        
                        for bitstring, count in counts.items():
                            for _ in range(min(count, 20)):
                                outcome = int(bitstring[0]) if bitstring else 0
                                tracker.add_measurement(outcome)
                
                if tracker.trajectory and len(tracker.trajectory) > 256:
                    trajectories[label] = np.array(tracker.trajectory)
                    print(f"  軌跡生成: {len(tracker.trajectory)}点")
                    
            except Exception as e:
                print(f"  エラー: {filename} - {e}")
    
    return trajectories

def compute_fourier_spectrum(trajectory, name):
    """軌跡のフーリエスペクトルを計算"""
    print(f"\n--- {name} のフーリエ解析 ---")
    
    # 実部と虚部を分離
    real_part = trajectory.real
    imag_part = trajectory.imag
    
    # 軌跡の長さ
    N = len(trajectory)
    print(f"データ点数: {N}")
    
    # サンプリング周波数（仮想的に1Hzとする）
    fs = 1.0
    dt = 1.0 / fs
    
    # フーリエ変換
    if scipy_available:
        # scipyを使用した高度な解析
        
        # 実部のフーリエ変換
        freqs_real = fftfreq(N, dt)
        fft_real = fft(real_part)
        
        # 虚部のフーリエ変換
        freqs_imag = fftfreq(N, dt)
        fft_imag = fft(imag_part)
        
        # 複素軌跡全体のフーリエ変換
        fft_complex = fft(trajectory)
        freqs_complex = fftfreq(N, dt)
        
        # パワースペクトル
        power_real = np.abs(fft_real) ** 2
        power_imag = np.abs(fft_imag) ** 2
        power_complex = np.abs(fft_complex) ** 2
        
        # 正の周波数のみ取得
        positive_freqs = freqs_complex[:N//2]
        power_real_pos = power_real[:N//2]
        power_imag_pos = power_imag[:N//2]
        power_complex_pos = power_complex[:N//2]
        
    else:
        # numpyのみを使用した基本解析
        fft_real = np.fft.fft(real_part)
        fft_imag = np.fft.fft(imag_part)
        fft_complex = np.fft.fft(trajectory)
        
        freqs_real = np.fft.fftfreq(N, dt)
        freqs_imag = np.fft.fftfreq(N, dt)
        freqs_complex = np.fft.fftfreq(N, dt)
        
        power_real = np.abs(fft_real) ** 2
        power_imag = np.abs(fft_imag) ** 2
        power_complex = np.abs(fft_complex) ** 2
        
        positive_freqs = freqs_complex[:N//2]
        power_real_pos = power_real[:N//2]
        power_imag_pos = power_imag[:N//2]
        power_complex_pos = power_complex[:N//2]
    
    # スペクトル特性の計算
    
    # 主要周波数の検出
    dominant_indices = np.argsort(power_complex_pos)[-5:][::-1]  # 上位5つ
    dominant_freqs = positive_freqs[dominant_indices]
    dominant_powers = power_complex_pos[dominant_indices]
    
    # スペクトルエントロピー
    normalized_power = power_complex_pos / np.sum(power_complex_pos)
    normalized_power = normalized_power[normalized_power > 0]  # ゼロ除去
    spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power))
    
    # 平均周波数
    mean_frequency = np.sum(positive_freqs * power_complex_pos) / np.sum(power_complex_pos)
    
    # スペクトル帯域幅
    variance_freq = np.sum(((positive_freqs - mean_frequency) ** 2) * power_complex_pos) / np.sum(power_complex_pos)
    spectral_bandwidth = np.sqrt(variance_freq)
    
    # スペクトル重心
    spectral_centroid = np.sum(positive_freqs * power_complex_pos) / np.sum(power_complex_pos)
    
    # 結果をまとめ
    result = {
        'name': name,
        'length': N,
        'dominant_frequencies': dominant_freqs,
        'dominant_powers': dominant_powers,
        'spectral_entropy': spectral_entropy,
        'mean_frequency': mean_frequency,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_centroid': spectral_centroid,
        'total_power': np.sum(power_complex_pos),
        'max_power': np.max(power_complex_pos),
        'frequencies': positive_freqs,
        'power_real': power_real_pos,
        'power_imag': power_imag_pos,
        'power_complex': power_complex_pos
    }
    
    # 統計出力
    print(f"スペクトルエントロピー: {spectral_entropy:.4f}")
    print(f"平均周波数: {mean_frequency:.4f} Hz")
    print(f"スペクトル帯域幅: {spectral_bandwidth:.4f} Hz")
    print(f"スペクトル重心: {spectral_centroid:.4f} Hz")
    print(f"総パワー: {np.sum(power_complex_pos):.2e}")
    print(f"主要周波数: {dominant_freqs[:3]}")
    
    return result

def analyze_time_frequency(trajectory, name):
    """時間-周波数解析（スペクトログラム）"""
    if not scipy_available:
        print(f"{name}: Scipy未インストールのため時間-周波数解析をスキップ")
        return None
    
    # 実部と虚部を分離
    real_part = trajectory.real
    imag_part = trajectory.imag
    
    # スペクトログラム計算用のパラメータ
    fs = 1.0  # サンプリング周波数
    nperseg = min(len(trajectory) // 4, 64)  # ウィンドウサイズ
    
    if nperseg < 4:
        print(f"{name}: データ長が短すぎるため時間-周波数解析をスキップ")
        return None
    
    try:
        # 実部のスペクトログラム
        f_real, t_real, Sxx_real = sp_signal.spectrogram(real_part, fs, nperseg=nperseg)
        
        # 虚部のスペクトログラム
        f_imag, t_imag, Sxx_imag = sp_signal.spectrogram(imag_part, fs, nperseg=nperseg)
        
        return {
            'name': name,
            'f_real': f_real,
            't_real': t_real,
            'Sxx_real': Sxx_real,
            'f_imag': f_imag,
            't_imag': t_imag,
            'Sxx_imag': Sxx_imag
        }
    
    except Exception as e:
        print(f"{name}: スペクトログラム計算エラー - {e}")
        return None

def visualize_fourier_analysis(fourier_results, spectrogram_results):
    """フーリエ解析結果の可視化"""
    n_trajectories = len(fourier_results)
    
    if n_trajectories == 0:
        print("可視化するフーリエ解析結果がありません")
        return
    
    # 動的にサブプロット数を決定
    n_rows = min(3, n_trajectories)
    n_cols = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(fourier_results[:n_rows]):
        name = result['name']
        
        # 1. パワースペクトル（実部・虚部・複素）
        ax = axes[i, 0]
        freqs = result['frequencies']
        
        ax.semilogy(freqs, result['power_real'], 'b-', alpha=0.7, label='実部')
        ax.semilogy(freqs, result['power_imag'], 'r-', alpha=0.7, label='虚部')
        ax.semilogy(freqs, result['power_complex'], 'k-', linewidth=2, label='複素')
        
        # 主要周波数をマーク
        for freq in result['dominant_frequencies'][:3]:
            if freq >= 0 and freq <= freqs[-1]:
                ax.axvline(freq, color='orange', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('周波数 [Hz]')
        ax.set_ylabel('パワー (log scale)')
        ax.set_title(f'{name}: パワースペクトル')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 位相スペクトル
        ax = axes[i, 1]
        fft_complex = np.fft.fft(result['power_complex'])  # 簡易版
        phase = np.angle(fft_complex[:len(freqs)])
        
        ax.plot(freqs, phase, 'g-', linewidth=2)
        ax.set_xlabel('周波数 [Hz]')
        ax.set_ylabel('位相 [rad]')
        ax.set_title(f'{name}: 位相スペクトル')
        ax.grid(True, alpha=0.3)
        
        # 3. スペクトログラム（実部）
        ax = axes[i, 2]
        if spectrogram_results and i < len(spectrogram_results) and spectrogram_results[i]:
            spec_data = spectrogram_results[i]
            if spec_data:
                im = ax.pcolormesh(spec_data['t_real'], spec_data['f_real'], 
                                  10 * np.log10(spec_data['Sxx_real'] + 1e-10), 
                                  shading='gouraud', cmap='viridis')
                ax.set_xlabel('時間')
                ax.set_ylabel('周波数 [Hz]')
                ax.set_title(f'{name}: スペクトログラム (実部)')
                plt.colorbar(im, ax=ax, label='パワー [dB]')
            else:
                ax.text(0.5, 0.5, 'スペクトログラム\n計算不可', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{name}: スペクトログラム')
        else:
            ax.text(0.5, 0.5, 'スペクトログラム\nデータなし', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name}: スペクトログラム')
        
        # 4. スペクトル特性サマリー
        ax = axes[i, 3]
        ax.axis('off')
        
        # テキスト情報
        info_text = f"""スペクトル特性:
        
エントロピー: {result['spectral_entropy']:.3f}
平均周波数: {result['mean_frequency']:.3f} Hz
帯域幅: {result['spectral_bandwidth']:.3f} Hz
重心: {result['spectral_centroid']:.3f} Hz
総パワー: {result['total_power']:.2e}
最大パワー: {result['max_power']:.2e}

主要周波数:
{', '.join([f'{f:.3f}' for f in result['dominant_frequencies'][:3]])} Hz
        """
        
        ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 余った軸を非表示
    for i in range(n_trajectories, n_rows):
        for j in range(n_cols):
            axes[i, j].set_visible(False)
    
    plt.suptitle('実データのフーリエ解析とスペクトル特性', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fourier_spectral_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_spectral_characteristics(fourier_results):
    """スペクトル特性の比較分析"""
    if not fourier_results:
        print("比較するスペクトル解析結果がありません")
        return None
    
    # 特徴量を抽出
    spectral_features = []
    
    for result in fourier_results:
        feature = {
            'name': result['name'],
            'data_type': 'bell' if 'bell' in result['name'] else 'qv',
            'noise_level': 'clean' if 'bell' in result['name'] else 
                          'moderate' if 'moderate' in result['name'] else 'high',
            'length': result['length'],
            'spectral_entropy': result['spectral_entropy'],
            'mean_frequency': result['mean_frequency'],
            'spectral_bandwidth': result['spectral_bandwidth'],
            'spectral_centroid': result['spectral_centroid'],
            'total_power': result['total_power'],
            'max_power': result['max_power'],
            'dominant_freq_1': result['dominant_frequencies'][0] if len(result['dominant_frequencies']) > 0 else 0,
            'dominant_freq_2': result['dominant_frequencies'][1] if len(result['dominant_frequencies']) > 1 else 0,
            'dominant_freq_3': result['dominant_frequencies'][2] if len(result['dominant_frequencies']) > 2 else 0
        }
        spectral_features.append(feature)
    
    df = pd.DataFrame(spectral_features)
    
    print("\n=== スペクトル特性比較分析 ===")
    print(df.round(4))
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. スペクトルエントロピー vs 帯域幅
    ax = axes[0, 0]
    colors = ['lightblue' if dt == 'bell' else 'orange' if 'moderate' in name else 'red' 
              for dt, name in zip(df['data_type'], df['name'])]
    
    scatter = ax.scatter(df['spectral_entropy'], df['spectral_bandwidth'], 
                        c=df['total_power'], s=100, alpha=0.7, cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='総パワー')
    ax.set_xlabel('スペクトルエントロピー')
    ax.set_ylabel('スペクトル帯域幅')
    ax.set_title('エントロピー vs 帯域幅')
    
    for i, name in enumerate(df['name']):
        ax.annotate(name.replace('_', '\\n'), 
                    (df['spectral_entropy'][i], df['spectral_bandwidth'][i]), 
                    fontsize=8, ha='center')
    ax.grid(True, alpha=0.3)
    
    # 2. データタイプ別エントロピー比較
    ax = axes[0, 1]
    bell_entropy = df[df['data_type'] == 'bell']['spectral_entropy']
    qv_entropy = df[df['data_type'] == 'qv']['spectral_entropy']
    
    ax.hist(bell_entropy, alpha=0.7, label='Bell状態', bins=5, color='lightblue')
    ax.hist(qv_entropy, alpha=0.7, label='Quantum Volume', bins=5, color='orange')
    ax.set_xlabel('スペクトルエントロピー')
    ax.set_ylabel('頻度')
    ax.set_title('データタイプ別エントロピー分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 主要周波数の分布
    ax = axes[0, 2]
    freq_data = []
    labels = []
    
    for i, row in df.iterrows():
        for j in range(1, 4):
            freq_col = f'dominant_freq_{j}'
            if row[freq_col] > 0:
                freq_data.append(row[freq_col])
                labels.append(f"{row['name']}_f{j}")
    
    if freq_data:
        ax.hist(freq_data, bins=10, alpha=0.7, color='green')
        ax.set_xlabel('周波数 [Hz]')
        ax.set_ylabel('頻度')
        ax.set_title('主要周波数の分布')
        ax.grid(True, alpha=0.3)
    
    # 4. パワー特性の比較
    ax = axes[1, 0]
    x_pos = np.arange(len(df))
    width = 0.4
    
    ax.bar(x_pos - width/2, np.log10(df['total_power']), width, label='総パワー (log10)', alpha=0.8)
    ax.bar(x_pos + width/2, np.log10(df['max_power']), width, label='最大パワー (log10)', alpha=0.8)
    
    ax.set_xlabel('軌跡')
    ax.set_ylabel('パワー (log10)')
    ax.set_title('パワー特性の比較')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name.replace('_', '\\n') for name in df['name']], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 平均周波数 vs 重心周波数
    ax = axes[1, 1]
    ax.scatter(df['mean_frequency'], df['spectral_centroid'], 
               s=100, alpha=0.7, c=colors)
    
    # 対角線
    min_freq = min(df['mean_frequency'].min(), df['spectral_centroid'].min())
    max_freq = max(df['mean_frequency'].max(), df['spectral_centroid'].max())
    ax.plot([min_freq, max_freq], [min_freq, max_freq], 'k--', alpha=0.5)
    
    ax.set_xlabel('平均周波数')
    ax.set_ylabel('スペクトル重心')
    ax.set_title('平均周波数 vs スペクトル重心')
    ax.grid(True, alpha=0.3)
    
    # 6. データ長 vs スペクトル特性
    ax = axes[1, 2]
    scatter = ax.scatter(df['length'], df['spectral_entropy'], 
                        c=df['spectral_bandwidth'], s=100, alpha=0.7, cmap='plasma')
    plt.colorbar(scatter, ax=ax, label='スペクトル帯域幅')
    ax.set_xlabel('データ長')
    ax.set_ylabel('スペクトルエントロピー')
    ax.set_title('データ長 vs エントロピー')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('スペクトル特性の比較分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('spectral_characteristics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # CSVで保存
    df.to_csv('fourier_spectral_analysis_results.csv', index=False)
    print("\nスペクトル解析結果を fourier_spectral_analysis_results.csv に保存")
    
    return df

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("実データのフーリエ解析とスペクトル特性解析")
    print(f"実行開始: {datetime.now()}")
    print("=" * 60)
    
    # 1. データ読み込み
    print("\n1. フーリエ解析用データ読み込み中...")
    trajectories = load_trajectories_for_fourier()
    
    if not trajectories:
        print("エラー: 利用可能な軌跡データがありません")
        return
    
    print(f"\n総軌跡数: {len(trajectories)}")
    
    # 2. フーリエ解析
    print("\n2. フーリエ解析実行中...")
    fourier_results = []
    
    for name, trajectory in trajectories.items():
        result = compute_fourier_spectrum(trajectory, name)
        fourier_results.append(result)
    
    # 3. 時間-周波数解析
    print("\n3. 時間-周波数解析実行中...")
    spectrogram_results = []
    
    for name, trajectory in trajectories.items():
        spec_result = analyze_time_frequency(trajectory, name)
        spectrogram_results.append(spec_result)
    
    # 4. 可視化
    print("\n4. フーリエ解析結果の可視化中...")
    visualize_fourier_analysis(fourier_results, spectrogram_results)
    
    # 5. スペクトル特性比較
    print("\n5. スペクトル特性比較分析中...")
    spectral_df = compare_spectral_characteristics(fourier_results)
    
    # 6. 主要発見の報告
    print("\n" + "=" * 60)
    print("🔬 フーリエ解析の主要発見")
    print("=" * 60)
    
    if spectral_df is not None and len(spectral_df) > 0:
        bell_results = spectral_df[spectral_df['data_type'] == 'bell']
        qv_results = spectral_df[spectral_df['data_type'] == 'qv']
        
        print(f"\n📊 スペクトル特性統計:")
        
        if not bell_results.empty:
            bell_entropy_avg = bell_results['spectral_entropy'].mean()
            bell_bandwidth_avg = bell_results['spectral_bandwidth'].mean()
            print(f"\nBell状態データ:")
            print(f"  平均スペクトルエントロピー: {bell_entropy_avg:.4f}")
            print(f"  平均スペクトル帯域幅: {bell_bandwidth_avg:.4f}")
        
        if not qv_results.empty:
            qv_entropy_avg = qv_results['spectral_entropy'].mean()
            qv_bandwidth_avg = qv_results['spectral_bandwidth'].mean()
            print(f"\nQuantum Volumeデータ:")
            print(f"  平均スペクトルエントロピー: {qv_entropy_avg:.4f}")
            print(f"  平均スペクトル帯域幅: {qv_bandwidth_avg:.4f}")
        
        # 最も複雑なスペクトル
        max_entropy_idx = spectral_df['spectral_entropy'].idxmax()
        max_entropy_name = spectral_df.loc[max_entropy_idx, 'name']
        max_entropy_value = spectral_df.loc[max_entropy_idx, 'spectral_entropy']
        
        print(f"\n🌟 最も複雑なスペクトル:")
        print(f"  {max_entropy_name}: エントロピー = {max_entropy_value:.4f}")
        
        # 最も狭い帯域幅
        min_bandwidth_idx = spectral_df['spectral_bandwidth'].idxmin()
        min_bandwidth_name = spectral_df.loc[min_bandwidth_idx, 'name']
        min_bandwidth_value = spectral_df.loc[min_bandwidth_idx, 'spectral_bandwidth']
        
        print(f"\n📡 最も狭い帯域幅:")
        print(f"  {min_bandwidth_name}: 帯域幅 = {min_bandwidth_value:.4f} Hz")
        
        print(f"\n💡 科学的意義:")
        print(f"  - 実データの周波数領域特性を詳細に解析")
        print(f"  - Bell状態とノイズデータのスペクトル差異を定量化") 
        print(f"  - 複素軌跡の時間-周波数特性を可視化")
        print(f"  - 量子測定データの新たな解析手法を確立")
    
    print(f"\n実行完了: {datetime.now()}")
    print("生成されたファイル:")
    print("  - fourier_spectral_analysis_results.png")
    print("  - spectral_characteristics_comparison.png")
    print("  - fourier_spectral_analysis_results.csv")

if __name__ == "__main__":
    main()