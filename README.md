# Ultra-low-Power-VAD
An ultra-low power VAD (Voice Activity Detector) using Analog Feature extraction and Binarized Neural Network Classifier
1. **AFE.m**
- Code implementing Analog Feature Extraction using 16 channel Band Pass Fitler bank
- Labeling of features using Statistical VAD on integrated bispectrum of audio signal
- References 
  - [1]K. M. H. Badami, S. Lauwereins, W. Meert, and M. Verhelst, “A 90 nm CMOS, 6 μW power-proportional acoustic sensing frontend for voice activity detection,” IEEE J. Solid-State Circuits, vol. 51, no. 1, pp. 291–302, Jan. 2016.
  - [2]J. Ramirez, J. M. Gorriz, J. C. Segura, C. G. Puntonet and A. J. Rubio, "Speech/non-speech discrimination based on contextual information integrated bispectrum LRT," in IEEE Signal Processing Letters, vol. 13, no. 8, pp. 497-500, Aug. 2006.

2. **BNN_train.py**
- Code to train a Binarized Neural Network (BNN) to classify features into Speech/Non-speech, using Keras framework.
- References
  - [1]Courbariaux, M.; Bengio, Y. BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. arXiv 2016, arXiv:1602.02830.
