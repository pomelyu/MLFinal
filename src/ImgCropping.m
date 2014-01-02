function Croping_inst = ImgCropping( data_inst )
% map image to full imSize x imSize image

imSize = 80;
numData = size(data_inst, 1);
t = 0.2;
Croping_inst = zeros(numData, imSize * imSize);

for k = 1:numData
    oriImg = reshape(data_inst(k, :), 105, 122);
    
    %% Calcuate the border of threshold t
    % up
    up = 0;
    for i = 1:105
        oneRow = oriImg(i, :);
        oneRow = find(oneRow > t);
        if size(oneRow, 2) ~= 0
            up = i;
            break;
        end
    end
    
    % down
    down = 105;
    for i = 105:-1:1
        oneRow = oriImg(i, :);
        oneRow = find(oneRow > t);
        if size(oneRow, 2) ~= 0
            down = i;
            break;
        end
    end
    
    % left
    left = 0;
    for j = 1:122
        oneColumn = oriImg(:, j);
        oneColumn = find(oneColumn > t);
        if size(oneColumn, 1) ~= 0
            left = j;
            break;
        end
    end
    
    % right
    right = 122;
    for j = 122:-1:1
        oneColumn = oriImg(:, j);
        oneColumn = find(oneColumn > t);
        if size(oneColumn, 1) ~= 0
            right = j;
            break;
        end
    end
    
    clear lock oneRow oneColumn;
    
    %% map origin image to crop image
    xGrid = (right - left)/imSize;
    yGrid = (down - up)/imSize;
    
    for i = 1:imSize
        for j = 1:imSize
            x = left + (j-1) * xGrid;
            y = up   + (i-1) * yGrid;
            if x > 0 && x < 122 && y > 0 && y < 105
                Croping_inst(k, (i-1)*imSize + j) = oriImg(ceil(y), ceil(x));
            end
        end
    end
  
    Croping_inst = sparse(Croping_inst);
end


end

