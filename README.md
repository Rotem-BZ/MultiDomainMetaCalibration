# MultiDomainMetaCalibration

Confidence calibration – the problem of predicting probability estimates representative of the true correctness likelihood – is important for 
classification models in many applications. An existing paper called "Meta-Calibration: Learning of Model Calibration Using Differentiable
Expected Calibration Error" uses meta-learning techniques to train neural networks to be well-calibrated.
In this project, we extend that work to the setting of multiple domains; We define and implement a label-smoothing
meta learning technique for multiple domains, and test its calibration generalization quality in over-parameterized neural networks.
