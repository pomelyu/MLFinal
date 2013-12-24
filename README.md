MLFinal
=======

2013 NTU Machine Learning Final Project


== Directory ===============================================================
  ./src     :  source code
  ./result  :  temporary or final result
                  * save temporary result to reuse them in blending and bagging
  ./lib     :  library we use
                  * libsvm : for matlab2013 & osx10.9
  MLFinal.m :  main file

  PS: Put .dat in root directory

== Data ====================================================================
    % Data is an 1xN cell.
    % Data(1,i) is an data(D), which is Nx2 array
    % D(1,1) is the label of the D, D(1,2) is always zero
    % D(i,1) and D(i,2) is the pixel of character and the pixel value serparately 

    i.e.  [  label   0
            pixel1   pixel1_value
            pixel2   pixel2_value
               ...   ...            ]

    for example
        train.dat:
            2 123:0.2 234:0.5 456:0.7
           11  45:0.6  89:0.2

        trainData:
            cell{1,1} = [    2 0
                           123 0.2
                           234 0.5
                           456 0.7  ] 

            cell{1,2} = [   11 0
                            45 0.6
                            89 0.2  ]      
