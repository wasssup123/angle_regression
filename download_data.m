[XTrain,~,anglesTrain] = digitTrain4DArrayData;
[XTest,~,anglesTest]   = digitTest4DArrayData;
save('DigitsDataTrain.mat','XTrain','anglesTrain');
save('DigitsDataTest.mat','XTest','anglesTest');
