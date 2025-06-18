"""
量子計算の公開データセット収集モジュール（NumPy版）
CQTプロジェクト用にQiskitを使わずにNumPy + matplotlibで実装
"""
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class QuantumDataCollector:
    """量子データを体系的に収集するクラス（NumPy実装）"""
    
    def __init__(self, base_path: str = 'quantum_datasets'):
        self.base_path = base_path
        self.datasets = {}
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """データ保存用のディレクトリ構造を作成"""
        directories = [
            'raw_data',
            'processed_data',
            'bell_states',
            'vqe_results',
            'qaoa_landscapes',
            'error_characterization',
            'custom_experiments'
        ]
        for dir_name in directories:
            os.makedirs(os.path.join(self.base_path, dir_name), exist_ok=True)
    
    def collect_bell_state_data(self, shots: int = 8192) -> pd.DataFrame:
        """Bell状態の測定データを収集（NumPy実装）"""
        bell_states = {
            'phi_plus': self._simulate_phi_plus,
            'phi_minus': self._simulate_phi_minus,
            'psi_plus': self._simulate_psi_plus,
            'psi_minus': self._simulate_psi_minus
        }
        
        results = []
        for state_name, simulator_func in bell_states.items():
            outcomes = simulator_func(shots)
            counts = self._count_outcomes(outcomes)
            
            results.append({
                'state': state_name,
                'counts': counts,
                'shots': shots,
                'timestamp': datetime.now().isoformat()
            })
        
        df = pd.DataFrame(results)
        self._save_dataset(df, 'bell_states', 'bell_measurement_data')
        return df
    
    def collect_parametric_rotation_data(self, angles: Optional[np.ndarray] = None) -> pd.DataFrame:
        """パラメトリックな単一量子ビット回転データを収集"""
        if angles is None:
            angles = np.linspace(0, 2*np.pi, 50)
        
        data = []
        for theta in angles:
            # Y軸回転の確率計算: P(|1⟩) = sin²(θ/2)
            prob_1 = np.sin(theta/2)**2
            
            # 測定シミュレーション
            outcomes = np.random.choice([0, 1], size=1024, p=[1-prob_1, prob_1])
            counts_0 = np.sum(outcomes == 0)
            counts_1 = np.sum(outcomes == 1)
            
            data.append({
                'angle': theta,
                'counts_0': counts_0,
                'counts_1': counts_1,
                'probability_1': counts_1 / 1024,
                'theoretical_prob_1': prob_1
            })
        
        df = pd.DataFrame(data)
        self._save_dataset(df, 'custom_experiments', 'rotation_sweep_data')
        return df
    
    def collect_entanglement_data(self, n_qubits: int = 3, shots: int = 4096) -> pd.DataFrame:
        """多量子ビットエンタングル状態のデータを収集"""
        results = []
        
        # GHZ状態シミュレーション
        ghz_outcomes = self._simulate_ghz_state(n_qubits, shots)
        counts_ghz = self._count_outcomes(ghz_outcomes)
        
        results.append({
            'state_type': 'GHZ',
            'n_qubits': n_qubits,
            'counts': counts_ghz,
            'shots': shots
        })
        
        # W状態シミュレーション
        w_outcomes = self._simulate_w_state(n_qubits, shots)
        counts_w = self._count_outcomes(w_outcomes)
        
        results.append({
            'state_type': 'W',
            'n_qubits': n_qubits,
            'counts': counts_w,
            'shots': shots
        })
        
        df = pd.DataFrame(results)
        self._save_dataset(df, 'custom_experiments', f'entanglement_data_{n_qubits}qubits')
        return df
    
    def collect_noise_characterization_data(self, noise_levels: List[float] = None) -> pd.DataFrame:
        """ノイズ特性データを収集"""
        if noise_levels is None:
            noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1]
        
        results = []
        
        for noise_level in noise_levels:
            # ノイズありでBell状態をシミュレーション
            outcomes = self._simulate_phi_plus_with_noise(2048, noise_level)
            counts = self._count_outcomes(outcomes)
            fidelity = self._calculate_bell_fidelity(counts, 'phi_plus')
            
            results.append({
                'noise_level': noise_level,
                'counts': counts,
                'fidelity': fidelity
            })
        
        df = pd.DataFrame(results)
        self._save_dataset(df, 'error_characterization', 'noise_sweep_data')
        return df
    
    def collect_vqe_landscape_data(self, hamiltonian: str = 'H2') -> pd.DataFrame:
        """VQEエネルギーランドスケープデータを収集"""
        # パラメータスイープ
        theta_range = np.linspace(0, 2*np.pi, 20)
        phi_range = np.linspace(0, 2*np.pi, 20)
        
        landscape_data = []
        
        for theta in theta_range:
            for phi in phi_range:
                params = [theta, phi]
                
                # H2分子のエネルギー期待値を模擬計算
                energy = self._calculate_h2_energy(params)
                
                landscape_data.append({
                    'theta': theta,
                    'phi': phi,
                    'energy': energy,
                    'hamiltonian': hamiltonian
                })
        
        df = pd.DataFrame(landscape_data)
        self._save_dataset(df, 'vqe_results', f'vqe_landscape_{hamiltonian}')
        return df
    
    def _simulate_phi_plus(self, shots: int) -> np.ndarray:
        """Bell状態 |Φ+⟩ の測定をシミュレーション"""
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2 → 50%で|00⟩、50%で|11⟩
        outcomes = np.random.choice(['00', '11'], size=shots, p=[0.5, 0.5])
        return outcomes
    
    def _simulate_phi_minus(self, shots: int) -> np.ndarray:
        """Bell状態 |Φ-⟩ の測定をシミュレーション"""
        # |Φ-⟩ = (|00⟩ - |11⟩)/√2 → 50%で|00⟩、50%で|11⟩
        outcomes = np.random.choice(['00', '11'], size=shots, p=[0.5, 0.5])
        return outcomes
    
    def _simulate_psi_plus(self, shots: int) -> np.ndarray:
        """Bell状態 |Ψ+⟩ の測定をシミュレーション"""
        # |Ψ+⟩ = (|01⟩ + |10⟩)/√2 → 50%で|01⟩、50%で|10⟩
        outcomes = np.random.choice(['01', '10'], size=shots, p=[0.5, 0.5])
        return outcomes
    
    def _simulate_psi_minus(self, shots: int) -> np.ndarray:
        """Bell状態 |Ψ-⟩ の測定をシミュレーション"""
        # |Ψ-⟩ = (|01⟩ - |10⟩)/√2 → 50%で|01⟩、50%で|10⟩
        outcomes = np.random.choice(['01', '10'], size=shots, p=[0.5, 0.5])
        return outcomes
    
    def _simulate_phi_plus_with_noise(self, shots: int, noise_level: float) -> np.ndarray:
        """ノイズありでBell状態|Φ+⟩をシミュレーション"""
        # 理想的な結果
        ideal_outcomes = self._simulate_phi_plus(shots)
        
        # ノイズを追加（bit-flipエラー）
        noisy_outcomes = []
        for outcome in ideal_outcomes:
            if np.random.random() < noise_level:
                # ランダムにビットをフリップ
                if outcome == '00':
                    noisy_outcome = np.random.choice(['01', '10'])
                elif outcome == '11':
                    noisy_outcome = np.random.choice(['01', '10'])
                else:
                    noisy_outcome = outcome
                noisy_outcomes.append(noisy_outcome)
            else:
                noisy_outcomes.append(outcome)
        
        return np.array(noisy_outcomes)
    
    def _simulate_ghz_state(self, n_qubits: int, shots: int) -> np.ndarray:
        """GHZ状態の測定をシミュレーション"""
        # |GHZ⟩ = (|000...⟩ + |111...⟩)/√2
        all_zeros = '0' * n_qubits
        all_ones = '1' * n_qubits
        outcomes = np.random.choice([all_zeros, all_ones], size=shots, p=[0.5, 0.5])
        return outcomes
    
    def _simulate_w_state(self, n_qubits: int, shots: int) -> np.ndarray:
        """W状態の測定をシミュレーション"""
        # |W⟩ = (|100...⟩ + |010...⟩ + |001...⟩ + ...)/√n
        # 各位置に1つだけ|1⟩がある状態
        basis_states = []
        for i in range(n_qubits):
            state = '0' * n_qubits
            state = state[:i] + '1' + state[i+1:]
            basis_states.append(state)
        
        prob = 1.0 / n_qubits
        probs = [prob] * n_qubits
        outcomes = np.random.choice(basis_states, size=shots, p=probs)
        return outcomes
    
    def _count_outcomes(self, outcomes: np.ndarray) -> Dict[str, int]:
        """測定結果をカウント"""
        unique, counts = np.unique(outcomes, return_counts=True)
        return dict(zip(unique, counts))
    
    def _calculate_bell_fidelity(self, counts: Dict[str, int], bell_type: str) -> float:
        """Bell状態の忠実度を計算"""
        total = sum(counts.values())
        
        if bell_type in ['phi_plus', 'phi_minus']:
            # |00⟩ と |11⟩ が理想
            ideal_counts = counts.get('00', 0) + counts.get('11', 0)
        elif bell_type in ['psi_plus', 'psi_minus']:
            # |01⟩ と |10⟩ が理想
            ideal_counts = counts.get('01', 0) + counts.get('10', 0)
        else:
            ideal_counts = 0
        
        return ideal_counts / total if total > 0 else 0.0
    
    def _calculate_h2_energy(self, params: List[float]) -> float:
        """H2分子のエネルギーを模擬計算"""
        # 簡略化されたH2ハミルトニアンの期待値
        theta, phi = params
        
        # 実際のH2分子の最小エネルギーは約-1.137 hartree
        # パラメータ依存の模擬計算
        energy = -1.0 + 0.2 * np.sin(theta) * np.cos(phi) + 0.1 * np.cos(2*theta)
        
        return energy
    
    def _save_dataset(self, data: pd.DataFrame, category: str, name: str):
        """データセットを保存"""
        filepath = os.path.join(self.base_path, category, f"{name}.csv")
        data.to_csv(filepath, index=False)
        
        # メタデータも保存
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'shape': data.shape,
            'columns': list(data.columns),
            'category': category,
            'name': name,
            'implementation': 'NumPy_based'
        }
        
        metadata_path = os.path.join(self.base_path, category, f"{name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved: {filepath}")