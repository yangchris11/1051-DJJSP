% SVM Example 1 for LIBSVM 

% ��J LIBSVM (Matlab version) �Ҧb����Ƨ�
addpath('C:\Users\MD531\Documents\Machine Learning\libsvm-3.18');

% training data and labels
features_a=[125,160,180,176,135,155,167,173,154,170;
            28, 58, 75, 62, 34, 45, 73, 68, 56, 53]';
% first column: the first feature (height in cm)
% second column: the second feature (weight in kg)
label_a = [1,2,2,1,1,1,2,2,2,1]';
% 1: �����~�H, 2: ���~�H

% test data and labels
features_b=[145,176,179,158,163;
            28,  71, 58, 54, 50]';
label_b = [1,2,1,2,1]';

% scaling
[m,N]=size(features_a);
[m1,N]=size(features_b);
mf=mean(features_a);
nrm=diag(1./std(features_a,1));
features_1=(features_a-ones(m,1)*mf)*nrm;
features_2=(features_b-ones(m1,1)*mf)*nrm;
% SVM
model = svmtrain(label_a, features_1);
% test
[predicted, accuracy, d_values] = svmpredict(label_b, features_2, model);
% predicted: the SVM output of the test data

