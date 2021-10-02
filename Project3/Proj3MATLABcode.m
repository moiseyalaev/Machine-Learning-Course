%% MATH 156 (7/2)
% |Discriminant Functions|
% |We now turn to classification.  We will begin by generating data that comes 
% from two classes in order to understand how the least-squares classifier works.|

C1numpoints = 100;
C2numpoints = 100;

mu1 = [1 2]';
sigma1 = [3 1.5; 1.5 1];
C1 = mvnrnd(mu1,sigma1,C1numpoints);

mu2 = [1 -2]';
sigma2 = [3 1.5; 1.5 1];
C2 = mvnrnd(mu2,sigma2,C2numpoints);

plot(C1(:,1),C1(:,2),'bo',C2(:,1),C2(:,2),'r+')
axis([-4 6 -6 6])
%% 
% |Now, we collect the data into the form required for the least-squares classifier.|

Xtilde = [ones(C1numpoints,1) C1; ones(C2numpoints,1) C2];
T = [ones(C1numpoints,1) zeros(C1numpoints,1); zeros(C2numpoints,1) ones(C2numpoints,1)];

Wtilde = pinv(Xtilde)*T;
x = -4:0.01:6;
wdec = Wtilde(:,1) - Wtilde(:,2);
y = (-wdec(1,1) - wdec(2,1)*x)/wdec(3,1);

plot(C1(:,1),C1(:,2),'bo',C2(:,1),C2(:,2),'r+', x,y,'k-')
axis([-4 6 -6 6])
%% 
% We can now explore how robust the classifier is to different settings.  First 
% we explore when the data points come from different  distributions.

C1numpoints = 100;
C2numpoints = 100;

mu1 = [1 2]';
sigma1 = [3 1.5; 1.5 1];
C1 = mvnrnd(mu1,sigma1,C1numpoints);

mu2 = [1 -2]';
sigma2 = [2 1; 1 2];
C2 = mvnrnd(mu2,sigma2,C2numpoints);

Xtilde = [ones(C1numpoints,1) C1; ones(C2numpoints,1) C2];
T = [ones(C1numpoints,1) zeros(C1numpoints,1); zeros(C2numpoints,1) ones(C2numpoints,1)];

Wtilde = pinv(Xtilde)*T;
x = -4:0.01:6;
wdec = Wtilde(:,1) - Wtilde(:,2);
y = (-wdec(1,1) - wdec(2,1)*x)/wdec(3,1);

plot(C1(:,1),C1(:,2),'bo',C2(:,1),C2(:,2),'r+', x,y,'k-')
axis([-4 6 -6 6])
%% 
% Next we explore when the size of the data classes differ.

C1numpoints = 100;
C2numpoints = 10;

mu1 = [1 2]';
sigma1 = [3 1.5; 1.5 1];
C1 = mvnrnd(mu1,sigma1,C1numpoints);

mu2 = [1 -2]';
sigma2 = [2 1; 1 2];
C2 = mvnrnd(mu2,sigma2,C2numpoints);

Xtilde = [ones(C1numpoints,1) C1; ones(C2numpoints,1) C2];
T = [ones(C1numpoints,1) zeros(C1numpoints,1); zeros(C2numpoints,1) ones(C2numpoints,1)];

Wtilde = pinv(Xtilde)*T;
x = -4:0.01:6;
wdec = Wtilde(:,1) - Wtilde(:,2);
y = (-wdec(1,1) - wdec(2,1)*x)/wdec(3,1);

plot(C1(:,1),C1(:,2),'bo',C2(:,1),C2(:,2),'r+', x,y,'k-')
axis([-4 6 -6 6])
%% 
% When the data overlaps significantly...

C1numpoints = 100;
C2numpoints = 100;

mu1 = [1 2]';
sigma1 = [3 1.5; 1.5 1];
C1 = mvnrnd(mu1,sigma1,C1numpoints);

mu2 = [1 0]';
sigma2 = [2 1; 1 2];
C2 = mvnrnd(mu2,sigma2,C2numpoints);

Xtilde = [ones(C1numpoints,1) C1; ones(C2numpoints,1) C2];
T = [ones(C1numpoints,1) zeros(C1numpoints,1); zeros(C2numpoints,1) ones(C2numpoints,1)];

Wtilde = pinv(Xtilde)*T;
x = -4:0.01:6;
wdec = Wtilde(:,1) - Wtilde(:,2);
y = (-wdec(1,1) - wdec(2,1)*x)/wdec(3,1);

plot(C1(:,1),C1(:,2),'bo',C2(:,1),C2(:,2),'r+', x,y,'k-')
axis([-4 6 -6 6])
%% 
% And when one data class has outliers...

C1numpoints = 100;
C2numpoints = 100;

mu1 = [1 2]';
sigma1 = [3 1.5; 1.5 1];
C1 = mvnrnd(mu1,sigma1,C1numpoints);

mu2 = [1 -1]';
sigma2 = [3 1.5; 1.5 1];
C21 = mvnrnd(mu2,sigma2,C2numpoints/2);

mu3 = [5 -5]';
sigma3 = [0.5 0; 0 0.5];
C22 = mvnrnd(mu3,sigma3,C2numpoints/2);
C2 = [C21; C22];

Xtilde = [ones(C1numpoints,1) C1; ones(C2numpoints,1) C2];
T = [ones(C1numpoints,1) zeros(C1numpoints,1); zeros(C2numpoints,1) ones(C2numpoints,1)];

Wtilde = pinv(Xtilde)*T;
x = -4:0.01:6;
wdec = Wtilde(:,1) - Wtilde(:,2);
y = (-wdec(1,1) - wdec(2,1)*x)/wdec(3,1);

plot(C1(:,1),C1(:,2),'bo',C2(:,1),C2(:,2),'r+', x,y,'k-')
axis([-4 6 -6 6])