"""
v3エラー検出のテストスクリプト
改善された検出アルゴリズムの動作確認
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from cqt_tracker_v3 import OptimizedCQTTracker, optimized_error_detection
from noise_models import QuantumNoiseSimulator, NoiseParameters, NoiseType

def test_v3_detection():
    """v3検出アルゴリズムのテスト"""
    print("=== CQT v3 エラー検出テスト ===\n")
    
    results = []
    
    # テストケース1: クリーンな重ね合わせ状態
    print("テスト1: クリーンな重ね合わせ状態")
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    tracker1 = OptimizedCQTTracker()
    trajectory1 = []
    
    for i in range(100):
        prob_1 = np.abs(state[1])**2
        outcome = 1 if np.random.random() < prob_1 else 0
        z = tracker1.add_measurement(outcome, state)
        trajectory1.append(z)
    
    error1 = optimized_error_detection(trajectory1, "superposition")
    print(f"  検出結果: {error1}")
    results.append(("クリーン重ね合わせ", error1))
    
    # テストケース2: 弱いノイズ（5%脱分極）
    print("\nテスト2: 弱いノイズ（5%脱分極）")
    noise_params = NoiseParameters(NoiseType.DEPOLARIZING, 0.05)
    noise_sim = QuantumNoiseSimulator(noise_params)
    tracker2 = OptimizedCQTTracker()
    trajectory2 = []
    
    for i in range(100):
        noisy_state = noise_sim.apply_noise_to_state(state.copy())
        prob_1 = np.abs(noisy_state[1])**2
        outcome = 1 if np.random.random() < prob_1 else 0
        z = tracker2.add_measurement(outcome, noisy_state)
        trajectory2.append(z)
    
    error2 = optimized_error_detection(trajectory2, "superposition")
    print(f"  検出結果: {error2}")
    results.append(("5%脱分極", error2))
    
    # テストケース3: 中程度のノイズ（10%脱分極）
    print("\nテスト3: 中程度のノイズ（10%脱分極）")
    noise_params = NoiseParameters(NoiseType.DEPOLARIZING, 0.1)
    noise_sim = QuantumNoiseSimulator(noise_params)
    tracker3 = OptimizedCQTTracker()
    trajectory3 = []
    
    for i in range(100):
        noisy_state = noise_sim.apply_noise_to_state(state.copy())
        prob_1 = np.abs(noisy_state[1])**2
        outcome = 1 if np.random.random() < prob_1 else 0
        z = tracker3.add_measurement(outcome, noisy_state)
        trajectory3.append(z)
    
    error3 = optimized_error_detection(trajectory3, "superposition")
    print(f"  検出結果: {error3}")
    results.append(("10%脱分極", error3))
    
    # テストケース4: 強いノイズ（20%脱分極）
    print("\nテスト4: 強いノイズ（20%脱分極）")
    noise_params = NoiseParameters(NoiseType.DEPOLARIZING, 0.2)
    noise_sim = QuantumNoiseSimulator(noise_params)
    tracker4 = OptimizedCQTTracker()
    trajectory4 = []
    
    for i in range(100):
        noisy_state = noise_sim.apply_noise_to_state(state.copy())
        prob_1 = np.abs(noisy_state[1])**2
        outcome = 1 if np.random.random() < prob_1 else 0
        z = tracker4.add_measurement(outcome, noisy_state)
        trajectory4.append(z)
    
    error4 = optimized_error_detection(trajectory4, "superposition")
    print(f"  検出結果: {error4}")
    results.append(("20%脱分極", error4))
    
    # テストケース5: 振幅減衰ノイズ
    print("\nテスト5: 振幅減衰ノイズ（10%）")
    noise_params = NoiseParameters(NoiseType.AMPLITUDE_DAMPING, 0.1)
    noise_sim = QuantumNoiseSimulator(noise_params)
    tracker5 = OptimizedCQTTracker()
    trajectory5 = []
    
    for i in range(100):
        noisy_state = noise_sim.apply_noise_to_state(state.copy())
        prob_1 = np.abs(noisy_state[1])**2
        outcome = 1 if np.random.random() < prob_1 else 0
        z = tracker5.add_measurement(outcome, noisy_state)
        trajectory5.append(z)
    
    error5 = optimized_error_detection(trajectory5, "superposition")
    print(f"  検出結果: {error5}")
    results.append(("10%振幅減衰", error5))
    
    # テストケース6: ビット反転ノイズ
    print("\nテスト6: ビット反転ノイズ（5%）")
    noise_params = NoiseParameters(NoiseType.BIT_FLIP, 0.05)
    noise_sim = QuantumNoiseSimulator(noise_params)
    tracker6 = OptimizedCQTTracker()
    trajectory6 = []
    
    for i in range(100):
        noisy_state = noise_sim.apply_noise_to_state(state.copy())
        prob_1 = np.abs(noisy_state[1])**2
        outcome = 1 if np.random.random() < prob_1 else 0
        z = tracker6.add_measurement(outcome, noisy_state)
        trajectory6.append(z)
    
    error6 = optimized_error_detection(trajectory6, "superposition")
    print(f"  検出結果: {error6}")
    results.append(("5%ビット反転", error6))
    
    # テストケース7: 固有状態|0⟩
    print("\nテスト7: 固有状態|0⟩（クリーン）")
    eigenstate = np.array([1.0, 0.0])
    tracker7 = OptimizedCQTTracker()
    trajectory7 = []
    
    for i in range(100):
        z = tracker7.add_measurement(0, eigenstate)
        trajectory7.append(z)
    
    error7 = optimized_error_detection(trajectory7, "eigenstate")
    print(f"  検出結果: {error7}")
    results.append(("固有状態|0⟩", error7))
    
    # テストケース8: 固有状態にノイズ
    print("\nテスト8: 固有状態|0⟩（10%脱分極）")
    noise_params = NoiseParameters(NoiseType.DEPOLARIZING, 0.1)
    noise_sim = QuantumNoiseSimulator(noise_params)
    tracker8 = OptimizedCQTTracker()
    trajectory8 = []
    
    for i in range(100):
        noisy_state = noise_sim.apply_noise_to_state(eigenstate.copy())
        prob_1 = np.abs(noisy_state[1])**2
        outcome = 1 if np.random.random() < prob_1 else 0
        z = tracker8.add_measurement(outcome, noisy_state)
        trajectory8.append(z)
    
    error8 = optimized_error_detection(trajectory8, "eigenstate")
    print(f"  検出結果: {error8}")
    results.append(("固有状態+10%ノイズ", error8))
    
    # 結果サマリー
    print("\n=== 検出結果サマリー ===")
    detected = 0
    for test_name, result in results:
        status = "✓ 検出" if result != "NO_ERROR" else "✗ 未検出"
        print(f"{test_name:<20}: {result:<25} [{status}]")
        if result != "NO_ERROR":
            detected += 1
    
    print(f"\n検出率: {detected}/{len(results)} = {detected/len(results)*100:.1f}%")
    
    return results

def test_sensitivity_adjustment():
    """感度調整のテスト"""
    print("\n=== 感度調整テスト ===")
    
    # 中程度のノイズでテスト
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    noise_params = NoiseParameters(NoiseType.DEPOLARIZING, 0.08)
    noise_sim = QuantumNoiseSimulator(noise_params)
    
    tracker = OptimizedCQTTracker()
    trajectory = []
    
    for i in range(100):
        noisy_state = noise_sim.apply_noise_to_state(state.copy())
        prob_1 = np.abs(noisy_state[1])**2
        outcome = 1 if np.random.random() < prob_1 else 0
        z = tracker.add_measurement(outcome, noisy_state)
        trajectory.append(z)
    
    # 異なる感度でテスト
    sensitivities = [0.5, 0.8, 1.0, 1.2, 1.5]
    print("\n8%脱分極ノイズに対する感度別検出:")
    for sensitivity in sensitivities:
        error = optimized_error_detection(trajectory, "superposition", noise_sensitivity=sensitivity)
        print(f"  感度 {sensitivity}: {error}")

def main():
    """メイン実行"""
    # 基本テスト
    results = test_v3_detection()
    
    # 感度調整テスト
    test_sensitivity_adjustment()
    
    print("\n=== v3改善の効果 ===")
    print("✓ 相対的な変化に基づく検出により、ベースラインからの逸脱を検出")
    print("✓ 状態タイプに応じた動的な閾値設定")
    print("✓ ノイズ感度パラメータによる調整可能性")
    print("✓ より現実的な検出結果")

if __name__ == "__main__":
    main()