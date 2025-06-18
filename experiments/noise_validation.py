"""
CQT Noise Validation Experiment
CQTエラー検出の量子ノイズに対する性能検証実験

この実験では以下を検証します：
1. CQTトラッカーが異なる種類の量子ノイズを検出できるか
2. ノイズ強度とCQT検出感度の関係
3. 従来手法との性能比較
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
from datetime import datetime

from cqt_tracker_v2 import ImprovedCQTTracker, improved_error_detection
from noise_models import (QuantumNoiseSimulator, NoiseParameters, NoiseType, 
                         NoiseAnalyzer, create_noise_test_suite)

class CQTNoiseValidationExperiment:
    """CQTノイズ検証実験クラス"""
    
    def __init__(self, n_measurements: int = 1000):
        self.n_measurements = n_measurements
        self.results = {}
        
    def run_single_noise_test(self, state_type: str, noise_params: NoiseParameters) -> Dict:
        """
        単一ノイズ条件でのテスト
        
        Args:
            state_type: 量子状態タイプ ('eigenstate_0', 'superposition_plus', etc.)
            noise_params: ノイズパラメータ
            
        Returns:
            テスト結果辞書
        """
        # 理想状態の準備
        clean_state = self._prepare_quantum_state(state_type)
        
        # ノイズシミュレーター初期化
        noise_sim = QuantumNoiseSimulator(noise_params)
        
        # CQTトラッカー初期化（クリーン用とノイズ用）
        clean_tracker = ImprovedCQTTracker()
        noisy_tracker = ImprovedCQTTracker()
        
        # 測定実行
        clean_measurements = self._generate_clean_measurements(clean_state, self.n_measurements)
        noisy_measurements = noise_sim.generate_measurement_outcomes(clean_state, self.n_measurements)
        
        # CQT軌跡生成
        clean_trajectory = []
        noisy_trajectory = []
        
        for i in range(self.n_measurements):
            # クリーン軌跡
            clean_outcome, _ = clean_measurements[i]
            clean_z = clean_tracker.add_measurement(clean_outcome, clean_state)
            clean_trajectory.append(clean_z)
            
            # ノイズ軌跡
            noisy_outcome, noisy_state = noisy_measurements[i]
            noisy_z = noisy_tracker.add_measurement(noisy_outcome, noisy_state)
            noisy_trajectory.append(noisy_z)
            
        # CQTエラー検出テスト
        cqt_error_detected = self._test_cqt_error_detection(noisy_trajectory, state_type)
        
        # 軌跡比較解析
        trajectory_analysis = NoiseAnalyzer.compare_clean_vs_noisy_trajectories(
            clean_trajectory, noisy_trajectory)
        
        # ノイズ特徴解析
        noise_signature = NoiseAnalyzer.analyze_noise_signatures(noisy_trajectory)
        
        return {
            "state_type": state_type,
            "noise_type": noise_params.noise_type.value,
            "noise_strength": noise_params.strength,
            "cqt_error_detected": cqt_error_detected,
            "trajectory_analysis": trajectory_analysis,
            "noise_signature": noise_signature,
            "clean_trajectory": clean_trajectory,
            "noisy_trajectory": noisy_trajectory,
            "detection_accuracy": self._compute_detection_accuracy(cqt_error_detected, noise_params.strength)
        }
        
    def _prepare_quantum_state(self, state_type: str) -> np.ndarray:
        """量子状態の準備"""
        if state_type == "eigenstate_0":
            return np.array([1.0, 0.0])
        elif state_type == "eigenstate_1":
            return np.array([0.0, 1.0])
        elif state_type == "superposition_plus":
            return np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        elif state_type == "superposition_minus":
            return np.array([1/np.sqrt(2), -1/np.sqrt(2)])
        elif state_type == "superposition_i":
            return np.array([1/np.sqrt(2), 1j/np.sqrt(2)])
        else:
            return np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # デフォルト
            
    def _generate_clean_measurements(self, state: np.ndarray, n_measurements: int) -> List[Tuple[int, np.ndarray]]:
        """理想的な測定を生成"""
        prob_1 = np.abs(state[1])**2
        measurements = []
        
        for _ in range(n_measurements):
            outcome = 1 if np.random.random() < prob_1 else 0
            measurements.append((outcome, state.copy()))
            
        return measurements
        
    def _test_cqt_error_detection(self, trajectory: List[complex], state_type: str) -> str:
        """CQTエラー検出テスト"""
        # 状態タイプを変換
        if "eigenstate" in state_type:
            expected_state = "eigenstate"
        elif "superposition" in state_type:
            expected_state = "superposition"
        else:
            expected_state = "unknown"
            
        return improved_error_detection(trajectory, expected_state)
        
    def _compute_detection_accuracy(self, detection_result: str, noise_strength: float) -> float:
        """検出精度の計算"""
        # ノイズがある場合はエラーが検出されるべき
        has_noise = noise_strength > 0.01  # 1%以上のノイズ
        error_detected = detection_result != "NO_ERROR"
        
        # 正解率計算
        if has_noise and error_detected:
            return 1.0  # True Positive
        elif not has_noise and not error_detected:
            return 1.0  # True Negative
        elif has_noise and not error_detected:
            return 0.0  # False Negative
        else:
            return 0.0  # False Positive
            
    def run_comprehensive_validation(self) -> Dict:
        """包括的検証実験"""
        print("CQT Noise Validation Experiment 開始...")
        
        # テスト対象の状態
        test_states = ["eigenstate_0", "eigenstate_1", "superposition_plus", "superposition_minus"]
        
        # ノイズテストスイート
        noise_suite = create_noise_test_suite()
        
        all_results = []
        total_tests = len(test_states) * len(noise_suite)
        current_test = 0
        
        for state_type in test_states:
            print(f"\n状態タイプ: {state_type}")
            
            for noise_params in noise_suite:
                current_test += 1
                print(f"  テスト {current_test}/{total_tests}: {noise_params.noise_type.value} "
                      f"(強度: {noise_params.strength:.2f})")
                
                try:
                    result = self.run_single_noise_test(state_type, noise_params)
                    all_results.append(result)
                except Exception as e:
                    print(f"    エラー: {e}")
                    continue
                    
        # 結果まとめ
        summary = self._analyze_validation_results(all_results)
        
        self.results = {
            "experiment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "n_measurements": self.n_measurements,
                "total_tests": len(all_results)
            },
            "individual_results": all_results,
            "summary": summary
        }
        
        print(f"\n実験完了: {len(all_results)}個のテストが実行されました")
        return self.results
        
    def _analyze_validation_results(self, results: List[Dict]) -> Dict:
        """検証結果の分析"""
        if not results:
            return {"error": "No results to analyze"}
            
        # ノイズタイプ別の検出性能
        noise_type_performance = {}
        for noise_type in NoiseType:
            type_results = [r for r in results if r["noise_type"] == noise_type.value]
            if type_results:
                accuracies = [r["detection_accuracy"] for r in type_results]
                noise_type_performance[noise_type.value] = {
                    "mean_accuracy": np.mean(accuracies),
                    "std_accuracy": np.std(accuracies),
                    "detection_rate": sum(r["cqt_error_detected"] != "NO_ERROR" for r in type_results) / len(type_results)
                }
                
        # ノイズ強度vs検出率
        strength_analysis = {}
        strengths = sorted(set(r["noise_strength"] for r in results))
        for strength in strengths:
            strength_results = [r for r in results if r["noise_strength"] == strength]
            accuracies = [r["detection_accuracy"] for r in strength_results]
            detection_rate = sum(r["cqt_error_detected"] != "NO_ERROR" for r in strength_results) / len(strength_results)
            
            strength_analysis[strength] = {
                "mean_accuracy": np.mean(accuracies),
                "detection_rate": detection_rate,
                "n_tests": len(strength_results)
            }
            
        # 全体統計
        all_accuracies = [r["detection_accuracy"] for r in results]
        overall_detection_rate = sum(r["cqt_error_detected"] != "NO_ERROR" for r in results) / len(results)
        
        return {
            "overall_statistics": {
                "mean_accuracy": np.mean(all_accuracies),
                "std_accuracy": np.std(all_accuracies),
                "overall_detection_rate": overall_detection_rate,
                "total_tests": len(results)
            },
            "noise_type_performance": noise_type_performance,
            "strength_vs_detection": strength_analysis
        }
        
    def save_results(self, filename: str = None):
        """結果をファイルに保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cqt_noise_validation_{timestamp}.json"
            
        # 複素数をシリアライズ可能な形式に変換
        serializable_results = self._make_serializable(self.results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
        print(f"結果を {filename} に保存しました")
        
    def _make_serializable(self, obj):
        """複素数を含むオブジェクトをJSON serializable形式に変換"""
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.number):
            return float(obj)
        else:
            return obj
            
    def plot_validation_results(self, save_path: str = None):
        """検証結果をプロット"""
        if not self.results:
            print("プロット用の結果がありません。先に実験を実行してください。")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ノイズタイプ別検出性能
        self._plot_noise_type_performance(axes[0, 0])
        
        # 2. ノイズ強度vs検出率
        self._plot_strength_vs_detection(axes[0, 1])
        
        # 3. 軌跡例（最も効果的なノイズ）
        self._plot_trajectory_examples(axes[1, 0])
        
        # 4. ROC曲線近似
        self._plot_roc_approximation(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"プロットを {save_path} に保存しました")
        else:
            plt.show()
            
    def _plot_noise_type_performance(self, ax):
        """ノイズタイプ別性能プロット"""
        summary = self.results["summary"]
        noise_performance = summary["noise_type_performance"]
        
        noise_types = list(noise_performance.keys())
        detection_rates = [noise_performance[nt]["detection_rate"] for nt in noise_types]
        
        bars = ax.bar(noise_types, detection_rates, color='skyblue', alpha=0.7)
        ax.set_ylabel('Detection Rate')
        ax.set_title('CQT Error Detection by Noise Type')
        ax.set_ylim(0, 1)
        
        # 値をバーの上に表示
        for bar, rate in zip(bars, detection_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.2f}', ha='center', va='bottom')
                   
        ax.tick_params(axis='x', rotation=45)
        
    def _plot_strength_vs_detection(self, ax):
        """ノイズ強度vs検出率プロット"""
        summary = self.results["summary"]
        strength_analysis = summary["strength_vs_detection"]
        
        strengths = sorted([float(s) for s in strength_analysis.keys()])
        detection_rates = []
        for s in strengths:
            # 小数点精度の問題に対処
            key = str(s)
            if key not in strength_analysis:
                # 近似的なキーを探す
                for existing_key in strength_analysis.keys():
                    if abs(float(existing_key) - s) < 1e-6:
                        key = existing_key
                        break
            detection_rates.append(strength_analysis[key]["detection_rate"])
        
        ax.plot(strengths, detection_rates, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Noise Strength')
        ax.set_ylabel('Detection Rate')
        ax.set_title('Detection Rate vs Noise Strength')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
    def _plot_trajectory_examples(self, ax):
        """軌跡例のプロット"""
        # 最も検出しやすいノイズの例を選択
        best_result = None
        best_detectability = 0
        
        for result in self.results["individual_results"]:
            if "trajectory_analysis" in result:
                detectability = result["trajectory_analysis"].get("noise_detectability", 0)
                if detectability > best_detectability:
                    best_detectability = detectability
                    best_result = result
                    
        if best_result:
            clean_traj = best_result["clean_trajectory"]
            noisy_traj = best_result["noisy_trajectory"]
            
            # 複素軌跡をデシリアライズ
            if isinstance(clean_traj[0], dict):
                clean_traj = [complex(z["real"], z["imag"]) for z in clean_traj]
                noisy_traj = [complex(z["real"], z["imag"]) for z in noisy_traj]
                
            clean_real = [z.real for z in clean_traj[:100]]  # 最初の100点
            clean_imag = [z.imag for z in clean_traj[:100]]
            noisy_real = [z.real for z in noisy_traj[:100]]
            noisy_imag = [z.imag for z in noisy_traj[:100]]
            
            ax.plot(clean_real, clean_imag, 'b-', alpha=0.7, label='Clean', linewidth=2)
            ax.plot(noisy_real, noisy_imag, 'r-', alpha=0.7, label='Noisy', linewidth=2)
            
            ax.set_xlabel('Real Part')
            ax.set_ylabel('Imaginary Part')
            ax.set_title(f'Trajectory Comparison\n{best_result["noise_type"]} (strength: {best_result["noise_strength"]:.2f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
    def _plot_roc_approximation(self, ax):
        """ROC曲線近似"""
        # 異なるノイズ強度での結果からROC曲線を近似
        strengths = []
        tpr_values = []  # True Positive Rate
        fpr_values = []  # False Positive Rate
        
        summary = self.results["summary"]
        strength_analysis = summary["strength_vs_detection"]
        
        for strength_str, data in strength_analysis.items():
            strength = float(strength_str)
            detection_rate = data["detection_rate"]
            
            # 簡単なTPR/FPR近似（実際のROC曲線は更に詳細な分析が必要）
            if strength < 0.02:  # 低ノイズ
                tpr = detection_rate * 0.5  # 保守的な推定
                fpr = detection_rate * 0.3
            else:  # 高ノイズ
                tpr = detection_rate
                fpr = (1 - detection_rate) * 0.1
                
            tpr_values.append(tpr)
            fpr_values.append(fpr)
            
        # ROC曲線プロット
        sorted_pairs = sorted(zip(fpr_values, tpr_values))
        fpr_sorted, tpr_sorted = zip(*sorted_pairs)
        
        ax.plot(fpr_sorted, tpr_sorted, 'go-', linewidth=2, markersize=6)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Approximation')
        ax.legend()
        ax.grid(True, alpha=0.3)

def main():
    """メイン実行関数"""
    print("=== CQT Noise Validation Experiment ===")
    
    # 実験インスタンス作成
    experiment = CQTNoiseValidationExperiment(n_measurements=500)  # 実行時間短縮のため500測定
    
    # 実験実行
    results = experiment.run_comprehensive_validation()
    
    # 結果保存
    experiment.save_results()
    
    # 結果プロット
    experiment.plot_validation_results("cqt_noise_validation_plots.png")
    
    # 簡単なサマリー表示
    print("\n=== 実験サマリー ===")
    summary = results["summary"]["overall_statistics"]
    print(f"全体精度: {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}")
    print(f"検出率: {summary['overall_detection_rate']:.3f}")
    print(f"実行テスト数: {summary['total_tests']}")
    
    print("\nノイズタイプ別検出率:")
    noise_performance = results["summary"]["noise_type_performance"]
    for noise_type, performance in noise_performance.items():
        print(f"  {noise_type}: {performance['detection_rate']:.3f}")

if __name__ == "__main__":
    main()