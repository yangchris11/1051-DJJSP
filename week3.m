% SVM TESTING 1


training_a = csvread('training_data_1.csv') ;
testing_b = csvread('testing_data_1.csv') ;

label_a = csvread('training_label_1.csv') ;
label_b = csvread('testing_label_1.csv') ;

[m1,N] = size( training_a ) ;
[m2,N] = size( testing_b ) ;

mf = mean( training_a ) ;
nrm = diag( 1./std(training_a,1) ) ;

feature_a = ( training_a - ones(m1,1) * mf ) * nrm ;
feature_b = ( testing_b - ones(m2,1) * mf ) * nrm ;

model = svmtrain( label_a , feature_a ) ;

[ predicted , accuracy , d_values ] = svmpredict( label_b , feature_b , model ) ;


% optimization finished, #iter = 121
% nu = 0.221721
% obj = -38.771233, rho = 0.486849
% nSV = 58, nBSV = 49
% Total nSV = 58

>> [predicted, accuracy, d_values] = svmpredict(label_b, features_2, model);
Accuracy = 88.0952% (37/42) (classification)


