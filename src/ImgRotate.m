function [ outImg ] = ImgRotate( inImg )

W = size(inImg,1);
C = floor([W/2 W/2]);

tmpImg = zeros(W,W);

%% find the four bounder point
tmp = find(inImg(:,1) > 0);
left  = [1, tmp(1,1)];

tmp = find(inImg(:,W) > 0);
right = [W, tmp(1,1)];

tmp = find(inImg(1,:) > 0);
up    = [tmp(1,1), 1];

tmp = find(inImg(W,:));
down  = [tmp(1,1), W];

%% calculate rotate angle
vec1 = left - right;
vec2 = up - down;
vecN = vec1 + vec2;

theta = acos(dot(vecN, [-1 0])/norm(vecN));

if (vecN(1,2)) < 0
    theta = -theta;
end

%% rotate
for i = 1:W
    for j= 1:W
        tmp = [i j] * [cos(theta) -sin(theta); sin(theta) cos(theta)];
        tmp = round(tmp);
        if 0 < tmp(1,1) && tmp(1,1) <= W && 0 < tmp(1,2) && tmp(1,2) <= W
            tmpImg(i,j) = inImg(tmp(1,1), tmp(1,2));
        end
    end
end

%% crop
[Row, Col] = find(tmpImg > 0);
up    = min([Row; 122]);
down  = max([Row;   0]);
left  = min([Col; 105]);
right = max([Col;   0]);

%% map origin image to crop image
xGrid = (right - left)/W;
yGrid = (down - up)/W;

outImg = zeros(W,W);

for i = 1:W
    for j = 1:W
        x = left + (j-1) * xGrid;
        y = up   + (i-1) * yGrid;
        if (x > 0) && (x <= W) && (y > 0) && (y <= W)
            outImg(i, j) = tmpImg(ceil(y), ceil(x));
        end
    end
end

end