import json
import numpy as np
from .memory import StateMemoryCache
from .agent_lgb import AgentLightGBM
from .agent_tcn import AgentTCN
from .agent_bilstm import AgentBiLSTM
from .agent_trans import AgentTransformer

class MASCoordinator:
    def __init__(self, client):
        self.client = client
        self.memory = StateMemoryCache(similarity_threshold=0.05)
        
        self.expert_team = {
            "LightGBM": AgentLightGBM(client),
            "TCN": AgentTCN(client),
            "BiLSTM": AgentBiLSTM(client),
            "Transformer": AgentTransformer(client)
        }

    def conduct_debate_and_decide(self, preds_dict, macro_f):
        """执行认知共识演化流程 (Cognitive Consensus Evolution)"""
        # 物理与认知多维感知
        trend_slope, variance, kurtosis, rms_energy = macro_f[0], macro_f[1], macro_f[11], macro_f[12]
        pred_divergence = np.std(list(preds_dict.values()))
        
        print(f"\n" + "="*80)
        print(f" [MAS Cognitive Consensus Evolution Mode]")
        print(f" Physical Sensing: Gradient {trend_slope:.4f} | Volatility {variance:.4f} | Kurtosis {kurtosis:.4f}")
        print(f" Epistemic State: Prediction Divergence = {pred_divergence:.4f}")
        print("="*80)

        # 0. 认知记忆体检索
        cached_state = self.memory.search_similar_state(macro_f, current_preds_dict=preds_dict)
        if cached_state:
            weights = cached_state['weights']
            final_rul = sum(w * p for w, p in zip(weights, list(preds_dict.values())))
            print(f" [!] Memory Cache Hit: Reusing optimized posterior weight distribution.")
            return final_rul, {"Round2_Log": "Memory_Cache_Hit", "Round3_Log": "Consensus_Retrieved_from_Cache"}

        # 1. Round 1: 独立评估
        print("\n [Phase I: Independent Epistemic Evaluation]")
        round1_reports = {}
        for name, agent in self.expert_team.items():
            report = agent.react_diagnosis(preds_dict[name], macro_f)
            round1_reports[name] = report
            print(f"  - {name}: Pred {preds_dict[name]:.1f} | Conf {report.get('confidence_score', 0.5)}")

        peer_context = "\n".join([f"[{n}] Pred:{preds_dict[n]:.1f}, Conf:{r.get('confidence_score',0.5)}, Logic:{r.get('epistemic_justification','')}" for n, r in round1_reports.items()])

        # 2. Round 2: 信念修正
        print("\n [Phase II: Cross-Perspective Belief Revision]")
        round2_reports = {}
        r2_log_lines = []
        for name, agent in self.expert_team.items():
            others_context = "\n".join([line for line in peer_context.split('\n') if not line.startswith(f"[{name}]")])
            r2_report = agent.cross_reflection(preds_dict[name], macro_f, round1_reports[name], others_context)
            round2_reports[name] = r2_report
            
            old_c, new_c = round1_reports[name].get('confidence_score', 0.5), r2_report.get('posterior_confidence', 0.5)
            status = "⬇️ Downward Revision" if new_c < old_c else ("⬆️ Consistency Maintained" if new_c > old_c else "➖ Stable")
            print(f"  - {name} Validation Complete | Confidence: {old_c} -> {new_c} ({status})")
            r2_log_lines.append(f"[{name}] Confidence {old_c}->{new_c} | Critique: {r2_report.get('peer_critique', 'N/A')}")

        # 3. Round 3: 元聚合决策 (Meta-Decision)
        print("\n [Phase III: Meta-Decision Synthesis]")
        final_decision = self._coordinator_arbitration(preds_dict, round2_reports, macro_f)
        
        print(f"  Arbitration Logic: {final_decision.get('arbitration_reasoning', 'N/A')}")
        
        raw_weights = final_decision.get("final_weights", [0.25, 0.25, 0.25, 0.25])
        weights = [w / (sum(raw_weights) + 1e-8) for w in raw_weights]
            
        final_rul = sum(w * p for w, p in zip(weights, list(preds_dict.values())))
        print(f"\n [Consensus Achieved] Final Fused RUL: {final_rul:.2f} Cycles\n" + "="*80)
        
        self.memory.add_memory(macro_f, weights, final_rul, base_preds_dict=preds_dict)
        
        return final_rul, {
            "Round2_Log": "\n".join(r2_log_lines),
            "Round3_Log": final_decision.get('arbitration_reasoning', '')
        }

    def _coordinator_arbitration(self, preds_dict, r2_reports, macro_f):
        """基于非对称成本优化与架构兼容性的元决策协议"""
        trend_slope, variance, kurtosis, rms_energy = macro_f[0], macro_f[1], macro_f[11], macro_f[12]
        pred_divergence = np.std(list(preds_dict.values()))

        summary = "\n".join([f"- {n}: Prediction {preds_dict[n]:.1f}, Posterior Confidence {r2_reports[n].get('posterior_confidence', 0.5)}. Critique: {r2_reports[n].get('peer_critique', '')}" for n in self.expert_team.keys()])
            
        prompt = f"""
        [Meta-Decision Protocol: Meta-Aggregator Implementation]
        Task: Compute optimal fusion weights based on the evolution of cognitive consensus.

        [Multi-dimensional Sensing Graph]
        - Degradation Gradient: {trend_slope:.4f} 
        - Physical Volatility: {variance:.4f} 
        - Transient Impulsive Index (Kurtosis): {kurtosis:.4f} 
        - MAS Epistemic Divergence: {pred_divergence:.4f}

        [Node Posterior States]
        {summary}
        
        【Asymmetric Risk Optimization Guidelines】
        1. Overestimation Penalty: In aviation safety-critical systems, predicting RUL > True RUL incurs exponential costs. When [Epistemic Divergence] is significant, prioritize expert nodes with conservative (lower) RUL estimations that provide rigorous justifications.
        2. Inductive Bias Matching: 
           - High Kurtosis: System exhibits transient impulsive degradation. Assign higher weights to nodes with robust local receptive fields (e.g., TCN).
           - High Energy & Low Volatility: System is in a stable accelerated degradation phase. Prioritize nodes with long-range temporal attention (e.g., Transformer).
        3. Confidence Reliability: Downweight nodes that exhibited significant downward revision of confidence, as this indicates architectural limitations in the current regime.

        Output JSON:
        {{
            "arbitration_reasoning": "(String) Rigorous academic justification for weight distribution, referencing the sensing graph and asymmetric risk factors.",
            "final_weights": [LightGBM_w, TCN_w, BiLSTM_w, Transformer_w] 
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {"final_weights": [0.25, 0.25, 0.25, 0.25], "arbitration_reasoning": "Communication_Fallback_to_Uniform_Weights"}