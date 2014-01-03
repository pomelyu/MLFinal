function [train_target]=generate_tt(trainlabel)

dim1 = size(trainlabel,1);
dim2 = max(trainlabel);
train_target=zeros(dim2,dim1);
for i=1:dim1
    if trainlabel(i,1)~=0
    train_target(trainlabel(i,1),i)=1;
    else
        continue;
    end
end

