/*

  Predict class label for a Test data with a trained model.

  Usage
  ------

  [ytest_est , dist] = NN_predict(Xtest , yproto_est , Wproto_est  , [lambda_est] , [options]);

  
  Inputs
  -------

  Xtest                                 Test data (d x Ntest)
  yproto_est                            Estimated prototypes labels  (1 x Nproto)
  Wproto_est                            Estimated prototypes weigths (d x Nproto)
  lambda_est                            Estimated Weigths factor  (d x 1). Default lambda_est = ones(d , 1);
  options 
          metric                        1 for euclidian distance (default), 2 for d4

  
  Outputs
  -------
  
  yproto_est                            Estimated labels  (1 x Ntest)
  dist                                  Distance between Xtest and Prototypes (Nproto x Ntest)


  To compile
  ----------


  mex  -g  -output NN_predict.dll NN_predict.c

  mex -f mexopts_intel10.bat -output NN_predict.dll NN_predict.c


  Example 1
  ---------
  
  close all
  load ionosphere
  Nproto_pclass                      = 4*ones(1 , length(unique(y)));
  
  options.epsilonk                   = 0.005;
  options.epsilonl                   = 0.001;
  options.epsilonlambda              = 10e-8;
  options.xi                         = 10;
  options.nb_iterations              = 5000;
  options.metric_method              = 1;
  options.shuffle                    = 1;
  options.updatelambda               = 1;

  options.method                     = 7;
  options.holding.rho                = 0.7;
  options.holding.K                  = 1;


  X                                  = normalize(X);
  [Itrain , Itest]                   = sampling(X , y , options);
  [Xtrain , ytrain , Xtest , ytest]  = samplingset(X , y , Itrain , Itest);

  

  [yproto_ini , Wproto_ini , lambda_ini]              = ini_proto(Xtrain , ytrain , Nproto_pclass);
  [yproto_est , Wproto_est  , lambda_est , E_GRLVQ]   = grlvq_model(Xtrain , ytrain , options , yproto_ini , Wproto_ini  , lambda_ini);
  [ytest_est , disttest]                              = NN_predict(Xtest , yproto_est , Wproto_est , lambda_est ,options);
  [ytrain_est , disttrain]                            = NN_predict(Xtrain , yproto_est , Wproto_est , lambda_est , options);

  Perftrain                          = perf_classif(ytrain , ytrain_est); 
  Perftest                           = perf_classif(ytest , ytest_est);;

  dktrain                            = min(disttrain(yproto==0 , :));
  dltrain                            = min(disttrain(yproto~=0 , :));
  nutrain                            = (dktrain - dltrain)./(dktrain + dltrain);
  [tptrain , fptrain]                = basicroc(ytrain , nutrain);

   
  dktest                             = min(disttest(yproto==0 , :));
  dltest                             = min(disttest(yproto~=0 , :));
  nutest                             = (dktest - dltest)./(dktest + dltest);
  [tptest , fptest]                  = basicroc(ytest , nutest);


  disp('Performances Train/Test')
  disp([Perftrain , Perftest])
  
  figure(1)
  plot(E_GRLVQ);
  title('E_{GRLVQ}(t)' , 'fontsize' , 12)
  
  figure(2)
  stem(lambda_est);
  title('\lambda' , 'fontsize' , 12)

  figure(3)
  plot(fptrain , tptrain , fptest , tptest , 'r' , 'linewidth'  , 2)
  xlabel('false positive rate');
  ylabel('true positive rate');
  title('ROC curve','fontsize' , 12);
  legend(['Train'] , ['Test'])



 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 04/09/2006

 Reference "A new Generalized LVQ Algorithm via Harmonic to Minimumm Distance Measure Transition", A.K. Qin, P.N. Suganthan and J.J. Liang,
 ---------  IEEE International Conference on System, Man and Cybernetics, 2004

*/


#include <math.h>
#include <mex.h>

typedef struct OPTIONS 
{
  int    metric_method; 
} OPTIONS; 

/*-------------------------------------------------------------------------------------------------------------- */

/* Function prototypes */

void glvq_predict(double * , double * , double * , double * , int , int  , int  , OPTIONS , double * , double *);

/*-------------------------------------------------------------------------------------------------------------- */


void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    double *Xtest , *Wproto , *yproto , *lambda;
	OPTIONS options = {1};
	double *ytest_est , *dist;
	double *tmp;
	int i , d , Ntest  , Nproto  ;
	mxArray *mxtemp;

    /* Input 1  Xtrain */
	
	Xtest         = mxGetPr(prhs[0]);		
	if( mxGetNumberOfDimensions(prhs[0]) !=2 )
	{
		mexErrMsgTxt("Xtest must be (d x Ntest)");	
	}
	d         = mxGetM(prhs[0]);	 
	Ntest     = mxGetN(prhs[0]);

	/* Input 2   yproto */
		
	yproto         = mxGetPr(prhs[1]);
	if((mxGetNumberOfDimensions(prhs[1]) != 2) )
	{	
		mexErrMsgTxt("yproto must be (1 x Nproto)");
	}
	Nproto     = mxGetN(prhs[1]);

	/* Input 3  Wproto */
	
	Wproto    = mxGetPr(prhs[2]);
 	if((mxGetNumberOfDimensions(prhs[2]) != 2) || (mxGetM(prhs[2]) != d) || (mxGetN(prhs[2]) != Nproto))
	{
		mexErrMsgTxt("Wproto must be (d x Nproto)");
	}

	/* Input 4   lambda */

	if (nrhs >= 4 && !mxIsEmpty(prhs[3]))
	{
		lambda    = mxGetPr(prhs[3]);
	}
	else
	{
		lambda    = (double *)malloc(d*sizeof(double));
		for (i = 0 ; i < d ; i++)
		{
			lambda[i] = 1.0;
		}
	}
	if ( (nrhs >= 5) && !mxIsEmpty(prhs[4]) )	
	{
		mxtemp                                   = mxGetField(prhs[4] , 0 , "metric_method");
		if(mxtemp != NULL)
		{
			tmp                                  = mxGetPr(mxtemp);
			options.metric_method                = (int) tmp[0];
		}		
	}

	/*------ Outputs ----- */

	plhs[0]               = mxCreateDoubleMatrix(1 , Ntest, mxREAL);
	ytest_est             = mxGetPr(plhs[0]);

	plhs[1]               = mxCreateDoubleMatrix(Nproto , Ntest, mxREAL);	
	dist                  = mxGetPr(plhs[1]);
	
	/*------ Main Call ----- */

	glvq_predict(Xtest , Wproto , yproto , lambda , d , Ntest , Nproto , options , ytest_est , dist);

	/*------ Free Memory ----- */

	if(nrhs < 4 || mxIsEmpty(prhs[3]))
	{
		free(lambda);
	}
}
/*-------------------------------------------------------------------------------------------------------------- */
void glvq_predict(double *Xtest , double *Wproto , double *yproto , double *lambda , int d , int Ntest , int Nproto , OPTIONS options , double *ytest_est , double *dist)				   
{
	int i , j , l , ld , id , ind, lNproto;
	double  disttmp , temp , dist_min , double_max = 1.79769313486231*10e307;
	
	if (options.metric_method)
	{
		for(l = 0 ; l < Ntest ; l++)	
		{
			ld       = l*d;
			lNproto  = l*Nproto;
			dist_min = double_max; 
			ind      = 0;
			for (i = 0 ; i < Nproto ; i++)
			{
				id      = i*d;
				disttmp = 0.0;
				for( j = 0 ; j < d ; j++)	
				{
					temp     = (Xtest[j + ld] - Wproto[j + id]);
					disttmp += (lambda[j]*temp*temp);	
				}
				dist[i + lNproto] = disttmp;
				if(disttmp < dist_min)	
				{
					dist_min = disttmp;
					ind      = i;	
				}	
			}
			ytest_est[l] = yproto[ind];	
		}	
	}
	else
	{
		for(l = 0 ; l < Ntest ; l++)	
		{
			ld       = l*d;
			lNproto  = l*Nproto;
			dist_min = double_max; 
			ind      = 0;
			for (i = 0 ; i < Nproto ; i++)
			{
				id      = i*d;
				disttmp = 0.0;
				for( j = 0 ; j < d ; j++)	
				{
					temp     = (Xtest[j + ld] - Wproto[j + id]);
					disttmp += (lambda[j]*temp*temp*temp*temp);	
				}
				dist[i + lNproto] = disttmp;
				if(disttmp < dist_min)	
				{
					dist_min = disttmp;
					ind      = i;	
				}	
			}
			ytest_est[l] = yproto[ind];	
		}		
	}	
}
/*-------------------------------------------------------------------------------------------------------------- */


