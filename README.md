# QoEforesight
Data-Driven QoE Inference For HTTP Adaptive Streaming

Nowadays, significant challenges in video streaming Quality of Experience (QoE) are often triggered by
dynamic network conditions, heterogeneous device capabilities, and evolving user expectations. Our research
work proposed QoE-foresight, a comprehensive and integrated framework for QoE inference with concept
drift detection and automatic self-healing features in adaptive video streaming. QoE-Foresight utilizes
multi-modal ensemble learning by training a ridge-regression QoE predictor on fused network, playback,
device, and content features, while integrating it with a Hybrid Drift Detection Engine (HDDE) that
combines Hoeffding’s Drift Detection Method with Adaptive windowing (HDDM-A) residual monitoring
with Uncertainty-Aware Distribution Tracking (UADF), as well as a Double Deep Q-Network (DDQN) based
self-healing controller. A combination of several public datasets and the achieved QoE prediction accuracy is
leveraged (𝑅2 ≈ 0.953) and outperforms state-of-the-art baselines while maintaining low inference latency
(approximately 0.12 ms per sample) and high operational uptime (around 99.4%). For drift detection, the
HDDE attains an F1-score of approximately 96.7% with an average detection latency of approximately
0.035 s, and the reinforcement-learning self-healing module reaches a success rate of approximately 91%
with a mean QoE recovery of +0.73 MOS in drift-affected scenarios. SHapley Additive exPlanations (SHAP)
is utilized to have better insight and decision on QoE drivers; video quality features such as PSNR, bitrate,
and frame rate contribute 42.3% of the total attribution, followed by network, device, and buffer-related
factors. From the ablation studies, the contribution of multi-modal fusion of approximately +12.4% with
improvement in 𝑅2 over single-modality baselines is discovered, along with HDDE contributing +8.7%, and
self-healing improving up to +15.2%. These findings indicate that QoE-foresight is a suitable and scalable
option for real-time deployment in challenging environments.
