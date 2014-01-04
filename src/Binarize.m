function [ output_inst ] = Binarize( input_inst )

threshold = sum(sum(input_inst))/sum(sum(input_inst > 0));
output_inst = +(input_inst > threshold);

end

