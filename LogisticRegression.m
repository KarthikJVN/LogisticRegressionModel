clc;
clear all;
AngerScore = [80,77,70,68,64,60,50,46,40,35,30,25];
SecondHeartAttack = [1,1,0,1,0,1,1,0,1,0,0,1];



X = SecondHeartAttack.';
Y = AngerScore.';
B = glmfit(Y,X,'binomial')

%pihat = mnrval(B,X)