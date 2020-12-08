# Stint Estimator for Assetto Corsa Competizione
Predictor that uses data exported from MoteC files, generated from Assetto Corsa Competizione, to predict lap times during a long stint.

Pytorch was used as a framework for training a LSTM type (Long short-term memory) Neural Network on data collected from several cars of the ACSR league (https://assettoargentina.wixsite.com/competizione). Most of the data that was used for training was generated with the 2019 model tires.

The predictor takes as input the first 10 laps of a CSV exported from the MoteC software and estimates the time per lap for the next 20 laps.