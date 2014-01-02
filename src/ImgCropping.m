function Croping_inst = ImgCropping( data_inst )
% map image to full imSize x imSize image

imSize = 60;
numData = size(data_inst, 1);
t = 0.2;
Croping_inst = zeros(numData, imSize * imSize);

rStr = '';
for k = 1:numData
    oriImg = reshape(data_inst(k, :), 105, 122);
    
    [Row, Col] = find(oriImg > t);
    up    = min([Row; 122]);
    down  = max([Row;   0]);
    left  = min([Col; 105]);
    right = max([Col;   0]);

    %% map origin image to crop image
    xGrid = (right - left)/imSize;
    yGrid = (down - up)/imSize;
    
    for i = 1:imSize
        for j = 1:imSize
            x = left + (j-1) * xGrid;
            y = up   + (i-1) * yGrid;
            if (x > 0) && (x < 122) && (y > 0) && (y < 105)
                Croping_inst(k, (i-1)*imSize + j) = oriImg(ceil(y), ceil(x));
            end
        end
    end
    
    %% Reveal progress
    msg = sprintf('-- Done %04d/%04d', k, numData);
    fprintf([rStr msg]);
    rStr = repmat(sprintf('\b'),1,length(msg));
end
fprintf('\n');

Croping_inst = sparse(Croping_inst);

end

