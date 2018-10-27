# 1. Function to calculate 12 mel frquency cepstarl coefficients(mfccc), 12 delta mfcc coefficients, 12 double-delta mfcc coefficients, 1 energy feature, 1 delta energy feature, 1 double-delta energy feature
# 2. input: 1. y- audio signal
# 3. input: 2. fs - sampling frequency
# 4. input: 3. window_length - length of the each frame of the audio signal
# 5. inout: 4. the  - length of the overlap window while creating the frames of the audio signal
# 6. output: 1. feature_matrix - feature matrix of the audio signal with 39 features 