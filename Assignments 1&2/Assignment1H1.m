close all
clear all
ds = tabularTextDatastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000); %read file and stores data
T = read(ds); %stores data in a vector
size(T); %size of vector
Alpha=.01; %learning rate
m=length(T{:,1}); %number of samples
U0=T{:,2}; %all rows in column 2(date)
%U=T{:,4:19}; %features from column 4 to 19
U=T{1:12964,4:7}; %features from column 4 to 7 and 60% of data for training
U2=T{12965:17285,4:7}; %features from column 4 to 7 and 20% of data for cross-validation
U4=T{17286:21607,4:7}; %features from column 4 to 7 and 20% of data for testing
U1=T{1:12964,3}; %price and 60% of data for training
U3=T{12965:17285,3}; %price and 20% of data for cross-validation
U5=T{17286:21607, 3}; %price and 20% of data for testing

P=T{1:12964,3}; %price
A=T{1:12964,20};
%for i=3:19
%   f = figure;
%   scatter(T{:,i},P) %plot features aganist the area
%end

%U2=U.^2; %squaring of area
X=[ones(12964,1) U U1 U.^2 U.^3]; %polynomial hypothesis (training)
X1=[ones(4321,1) U2 U3 U2.^2 U2.^3]; %polynomial hypothesis (cross-validation)
X2=[ones(4322,1) U4 U5 U4.^2 U4.^3]; %polynomial hypothesis (testing)

n=length(X(1,:)); %number of columns
for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w)); %scaling mean normalization
    end
end

%Y=T{:,3}/mean(T{:,3}); %price normalization?
Theta=zeros(n,1); %vector of zeros
k=1;

E(k)=(1/(2*m))*sum((X*Theta-U1).^2); %cost function

R=1;
while R==1
Alpha=Alpha*1;
Theta=Theta-(Alpha/m)*X'*(X*Theta-U1); %gradient descent function
k=k+1;
E(k)=(1/(2*m))*sum((X*Theta-U1).^2);
if E(k-1)-E(k)<0
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.000001;
    R=0;
end
end
E=(1/(2*m))*sum((X1*Theta-Y).^2);
plot(E)