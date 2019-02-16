clear all
close all
clc
ds = tabularTextDatastore('heart_DD.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds)
size(T);
Alpha=.01;

m=length(T{1:150,1});
n=length(T{151:200,1});
o=length(T{201:250,1});

U=T{1:150,1:13};
U1=T{151:200,1:13};
U2=T{201:250,1:13};
U3=T{1:150,14};
U4=T{151:200,14};
U5=T{201:250,14};

%for j=1:13
%    f = figure;
%    scatter(T{:,j},T{:,14})
%end

X=[ones(size(U,1),1) U];
X1=[ones(n,1) U1];
X2=[ones(o,1) U2];

[i,j]=size(U);

Theta=zeros(j+1,1);

h = 1 ./ (1 + exp(-(X*Theta)));
k=1;
E(k) =-(1/m)*sum(U3 .* log(h) + (1-U3) .* log(1-h))

grad = zeros(size(Theta, 1),1);
for i=1:size(grad)-1
    grad(i)= (1/m) * sum( (h - U3)' * U(:,i))
end

R=1;

while R==1
Alpha=Alpha*1;
Theta=Theta-(Alpha/m)*X'*(X*Theta-U3); %gradient descent function
h = 1 ./ (1 + exp(-(X*Theta)));
k=k+1;
E(k)=-(1/m)*sum(U3 .* log(h) + (1-U3) .* log(1-h));
if E(k-1)-E(k)<0
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.000001;
    R=0;
end
end
plot(E)
h = 1 ./ (1 + exp(-(X1*Theta)));
E(k)=-(1/m)*sum(U4 .* log(h) + (1-U4) .* log(1-h))