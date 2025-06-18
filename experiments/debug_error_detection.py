"""
エラー検出デバッグスクリプト
CQTエラー検出が機能しない原因を調査
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from cqt_tracker_v2 import ImprovedCQTTracker, improved_error_detection
from noise_models import QuantumNoiseSimulator, NoiseParameters, NoiseType

def debug_error_detection():
    """エラー検出のデバッグ"""
    print("=== CQTエラー検出デバッグ ===\n")
    
    # テストケース1: クリーンな重ね合わせ状態
    print("テスト1: クリーンな重ね合わせ状態")
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    tracker1 = ImprovedCQTTracker()
    trajectory1 = []
    
    for i in range(100):
        prob_1 = np.abs(state[1])**2
        outcome = 1 if np.random.random() < prob_1 else 0
        z = tracker1.add_measurement(outcome, state)
        trajectory1.append(z)
    
    # 軌跡の統計を表示
    traj_array = np.array(trajectory1[-20:])
    print(f"  軌跡統計（最後の20測定）:")
    print(f"    実部: 平均={np.mean(traj_array.real):.3f}, 標準偏差={np.std(traj_array.real):.3f}")
    print(f"    虚部: 平均={np.mean(traj_array.imag):.3f}, 標準偏差={np.std(traj_array.imag):.3f}")
    print(f"    絶対値: 平均={np.mean(np.abs(traj_array)):.3f}")
    
    error1 = improved_error_detection(trajectory1, "superposition")
    print(f"  エラー検出結果: {error1}")
    print()
    
    # テストケース2: 強いノイズ（20%脱分極）
    print("テスト2: 強いノイズ（20%脱分極）")
    noise_params = NoiseParameters(NoiseType.DEPOLARIZING, 0.2)
    noise_sim = QuantumNoiseSimulator(noise_params)
    tracker2 = ImprovedCQTTracker()
    trajectory2 = []
    
    for i in range(100):
        noisy_state = noise_sim.apply_noise_to_state(state.copy())
        prob_1 = np.abs(noisy_state[1])**2
        outcome = 1 if np.random.random() < prob_1 else 0
        z = tracker2.add_measurement(outcome, noisy_state)
        trajectory2.append(z)
    
    traj_array2 = np.array(trajectory2[-20:])
    print(f"  軌跡統計（最後の20測定）:")
    print(f"    実部: 平均={np.mean(traj_array2.real):.3f}, 標準偏差={np.std(traj_array2.real):.3f}")
    print(f"    虚部: 平均={np.mean(traj_array2.imag):.3f}, 標準偏差={np.std(traj_array2.imag):.3f}")
    print(f"    絶対値: 平均={np.mean(np.abs(traj_array2)):.3f}")
    
    error2 = improved_error_detection(trajectory2, "superposition")
    print(f"  エラー検出結果: {error2}")
    print()
    
    # テストケース3: 固有状態|0⟩
    print("テスト3: 固有状態|0⟩")
    eigenstate = np.array([1.0, 0.0])
    tracker3 = ImprovedCQTTracker()
    trajectory3 = []
    
    for i in range(100):
        z = tracker3.add_measurement(0, eigenstate)  # 常に0を測定
        trajectory3.append(z)
    
    traj_array3 = np.array(trajectory3[-20:])
    print(f"  軌跡統計（最後の20測定）:")
    print(f"    実部: 平均={np.mean(traj_array3.real):.3f}, 標準偏差={np.std(traj_array3.real):.3f}")
    print(f"    虚部: 平均={np.mean(traj_array3.imag):.3f}, 標準偏差={np.std(traj_array3.imag):.3f}")
    
    error3 = improved_error_detection(trajectory3, "eigenstate")
    print(f"  エラー検出結果: {error3}")
    print()
    
    # 閾値の分析
    print("=== 検出閾値の分析 ===")
    print("デコヒーレンス検出: magnitude < 0.3 かつ std < 0.1")
    print("位相ドリフト検出: std(phase_diff) > π/2")
    print("重ね合わせ崩壊: 虚部平均 < 0.5")
    print("固有状態エラー: 虚部平均 > 0.2")
    print()
    
    # 問題点の診断
    print("=== 診断結果 ===")
    if np.mean(traj_array.imag) < 0.5:
        print("問題: 重ね合わせ状態でも虚部が0.5未満になっている")
        print("  → 閾値が高すぎる可能性")
    
    if np.mean(np.abs(traj_array2)) > 0.3:
        print("問題: 強いノイズでも絶対値が0.3以上")
        print("  → デコヒーレンス検出の閾値が低すぎる")
    
    return trajectory1, trajectory2, trajectory3

def test_detection_with_various_thresholds():
    """異なる閾値でのテスト"""
    print("\n=== 閾値変更テスト ===")
    
    # ノイズありの軌跡を生成
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    noise_params = NoiseParameters(NoiseType.DEPOLARIZING, 0.1)
    noise_sim = QuantumNoiseSimulator(noise_params)
    tracker = ImprovedCQTTracker()
    trajectory = []
    
    for i in range(100):
        noisy_state = noise_sim.apply_noise_to_state(state.copy())
        prob_1 = np.abs(noisy_state[1])**2
        outcome = 1 if np.random.random() < prob_1 else 0
        z = tracker.add_measurement(outcome, noisy_state)
        trajectory.append(z)
    
    # 軌跡の特徴量を計算
    recent = trajectory[-20:]
    traj_array = np.array(recent)
    
    magnitudes = np.abs(traj_array)
    phases = np.angle(traj_array)
    phase_diff = np.diff(phases)
    phase_diff = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi
    
    print(f"軌跡の特徴量:")
    print(f"  絶対値（最後5点）: 平均={np.mean(magnitudes[-5:]):.3f}, 標準偏差={np.std(magnitudes[-5:]):.3f}")
    print(f"  位相差の標準偏差: {np.std(phase_diff):.3f} (π/2 = {np.pi/2:.3f})")
    print(f"  虚部の平均: {np.mean(traj_array.imag):.3f}")
    print(f"  実部の範囲: [{np.min(traj_array.real):.3f}, {np.max(traj_array.real):.3f}]")
    print(f"  虚部の範囲: [{np.min(traj_array.imag):.3f}, {np.max(traj_array.imag):.3f}]")

def main():
    """メイン実行"""
    # 基本デバッグ
    traj1, traj2, traj3 = debug_error_detection()
    
    # 閾値テスト
    test_detection_with_various_thresholds()
    
    print("\n=== 推奨される改善 ===")
    print("1. 虚部の閾値を現実的な値に調整（0.5 → 0.3）")
    print("2. デコヒーレンス検出の絶対値閾値を上げる（0.3 → 0.5）")
    print("3. 状態タイプに応じた動的な閾値設定")
    print("4. ノイズタイプ別の検出ロジック追加")

if __name__ == "__main__":
    main()