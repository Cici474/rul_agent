import json

class BaseExpertAgent:
    def __init__(self, name, persona_desc, client):
        self.name = name
        self.persona = persona_desc
        self.client = client

    def react_diagnosis(self, pred_val, macro_f):
        """Phase I: 基于 ReAct 范式的独立异构认知评估"""
        # 提取核心宏观指标
        trend_slope, variance, energy = macro_f[0], macro_f[1], macro_f[12]
        
        prompt = f"""
        [System Role Definition]
        Act as an independent heterogeneous evaluation node in a Multi-Agent System (MAS) for aviation engine Prognostic and Health Management (PHM). 
        Architectural Inductive Bias: {self.persona}

        [System State Observation]
        Macro-statistical indicators from the current sliding window:
        - Global Degradation Gradient: {trend_slope:.4f}
        - Physical Volatility (Variance): {variance:.4f}
        - Signal Degradation Energy (RMS): {energy:.4f}
        
        [Uncalibrated Prediction]
        Underlying numerical model RUL output: {pred_val:.1f} Cycles.

        [Task: Epistemic Reasoning]
        Execute a ReAct (Reasoning and Acting) diagnosis. Assess the theoretical alignment between the current degradation trajectory and your architectural inductive bias.
        
        Output strictly in JSON format:
        {{
            "degradation_characterization": "(String) Characterize the current physical degradation phase based on gradient and volatility.",
            "structural_compatibility": "(String) Evaluate the compatibility between the current data distribution and your model's feature extraction domain.",
            "epistemic_justification": "(String) Provide rigorous academic justification supporting the current RUL prediction.",
            "confidence_score": (Float 0.0-1.0) Initial epistemic confidence score.
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.4
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {"confidence_score": 0.5, "epistemic_justification": "API_Inference_Error"}

    def cross_reflection(self, my_pred, macro_f, my_round1_report, peer_reports_context):
        """Phase II: 基于异构证据采信的信念修正 (Belief Revision)"""
        prompt = f"""
        [Phase: Cross-Perspective Belief Revision]
        Node Identity: {self.name}. 
        Prior State: Prediction {my_pred:.1f} | Initial Confidence {my_round1_report.get('confidence_score', 0.5)}.

        [Heterogeneous Peer Evidence]
        Diagnostic reasoning from other expert nodes:
        {peer_reports_context}

        [Task: Posterior Calibration]
        Critically analyze peer evidence to perform belief revision (Self-Calibration):
        1. Identify if peer architectures captured critical features (e.g., transient anomalies vs. long-term dependencies) that your inductive bias may have filtered.
        2. Adjust your confidence score based on evidence consensus or valid divergence. Defend your stance if your architecture remains theoretically optimal for the current macro state.
        
        Output strictly in JSON format:
        {{
            "peer_critique": "(String) Rigorous critique of the most divergent peer model, addressing architectural limitations or acknowledging feature validity.",
            "calibrated_reasoning": "(String) Updated epistemological reasoning after assimilating heterogeneous evidence.",
            "posterior_confidence": (Float 0.0-1.0) Calibrated confidence score after cross-validation.
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.6
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {"posterior_confidence": 0.5, "peer_critique": "API_Inference_Error"}