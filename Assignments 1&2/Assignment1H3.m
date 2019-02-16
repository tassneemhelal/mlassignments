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
U=T{1:12964,4:7}; %features from column 4 to 7
U1=T{1:12964,20:21}; %area features

P=T{1:12964,3}; %price
A=T{1:12964,20};
%for i=3:19
%   f = figure;
%   scatter(T{:,i},P) %plot features aganist the area
%end

U2=U.^2; %squaring of area
X=[ones(m,1) U U1.^2 U.^2 U.^4]; %polynomial hypothesis (concatination)

n=length(X(1,:)); %number of columns
for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w)); %scaling mean normalization
    end
end

Y=T{:,3}/mean(T{:,3}); %price normalization?
Theta=zeros(n,1); %vector of zeros
k=1;

E(k)=(1/(2*m))*sum((X*Theta-Y).^2); %cost function

R=1;
while R==1
Alpha=Alpha*1;
Theta=Theta-(Alpha/m)*X'*(X*Theta-Y); %gradient descent function
k=k+1;
E(k)=(1/(2*m))*sum((X*Theta-Y).^2);
if E(k-1)-E(k)<0
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.000001;
    R=0;
end
end
plot(E)