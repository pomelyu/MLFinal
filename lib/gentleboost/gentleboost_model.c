
/*

  Gentle AdaBoost Classifier with two different weak-learners : Decision Stump and Perceptron.
  Multi-class problem is performed with the one-vs-all strategy.

  Usage
  ------

  model = gentleboost_model(X , y , [options]);

  
  Inputs
  -------

  X                                     Features matrix (d x N) in double precision
  y                                     Labels vector(1 x N) where y_i={1,...,M} and i=1,...,N.. If y represent binary labels vector then y_i={-1,1}.
  options
              weaklearner               Choice of the weak learner used in the training phase
			                            weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R
			                            weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(a,b)) = sigmoid(x ; a,b) in R
              T                         Number of weaklearners (default T = 100)
			  epsi                      Epsilon constant in the sigmoid function used in the perceptron (default epsi = 1.0)
              lambda                    Regularization parameter for the perceptron's weights update (default lambda = 1e-3)
			  max_ite                   Maximum number of iterations of the perceptron algorithm (default max_ite = 100)
              seed                      Seed number for internal random generator (default random seed according to time)

If compiled with the "OMP" compilation flag
             num_threads                Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)

  Outputs
  -------
  
  model                                 Structure of model ouput
  
	          featureIdx                Features index in single/double precision of the T best weaklearners (T x m) where m is the number of class. 
			                            For binary classification m is force to 1.
			  th                        Optimal Threshold parameters (1 x T) in single/double precision.
			  a                         Affine parameter(1 x T) in single/double precision.
			  b                         Bias parameter (1 x T) in single/double precision.
			  weaklearner               Choice of the weak learner used in the training phase in single/double precision.
			  epsi                      Epsilon constant in the sigmoid function used in the perceptron in single/double precision.


  To compile
  ----------


  mex  gentleboost_model.c

  mex  -output gentleboost_model.dll gentleboost_model.c

  mex  -f mexopts_intel10.bat -output gentleboost_model.dll gentleboost_model.c

  If OMP directive is added, OpenMP support for multicore computation

  mex  -v -DOMP -f mexopts_intel10.bat -output gentleboost_model.dll gentleboost_model.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

  mex  -g -DOMP -f mexopts_intel10.bat -output gentleboost_model.dll gentleboost_model.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


  Example 1
  ---------

%  load iris %wine
  load wine
  %load heart
  %load wbc
  %y(y==0)                 = -1;

  options.weaklearner     = 0;
  options.epsi            = 0.5;
  options.lambda          = 1e-3;
  options.T               = 12;
  options.max_ite         = 3000;
  options.seed            = 1234543;
  
  model                   = gentleboost_model(X , y , options);
  [yest , fx]             = gentleboost_predict(X , model);
  Perf                    = sum(y == yest)/length(y)
  plot(fx')



  Example 2
  ---------

  clear
  load wine

  X                       = single(X);
  y                       = single(y);

  options.weaklearner     = 0;
  options.epsi            = 0.5;
  options.lambda          = 1e-3;
  options.T               = 12;
  options.max_ite         = 3000;
  options.seed            = 1234543;
  
  model                   = gentleboost_model(X , y , options);
  [yest , fx]             = gentleboost_predict(X , model);
  Perf                    = sum(y == yest)/length(y)
  plot(fx')


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009

 References   [1] Friedman, J. H., Hastie, T. and Tibshirani, R. "Additive Logistic Regression: a Statistical View of Boosting." (Aug. 1998) 
 ----------   [2] R.E Schapire and al "Boosting the margin : A new explanation for the effectiveness of voting methods". 
                  The annals of statistics, 1999 
*/ 


#include <time.h>
#include <math.h>
#include <mex.h>
#ifdef OMP
 #include <omp.h>
#endif

#define znew   (z = 36969*(z&65535) + (z>>16) )
#define wnew   (w = 18000*(w&65535) + (w>>16) )
#define MWC    ((znew<<16) + wnew )
#define SHR3   ( jsr ^= (jsr<<17), jsr ^= (jsr>>13), jsr ^= (jsr<<5) )
#define randint SHR3
#define rand() (0.5 + (signed)randint*2.328306e-10)

#ifdef __x86_64__
    typedef int UL;
#else
    typedef unsigned long UL;
#endif
static UL jsrseed = 31340134 , jsr;

#define huge 1e300
#define sign(a) ((a) >= (0) ? (1.0) : (-1.0))

#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

struct opts
{
  int    weaklearner;
  int    T;
  double epsi;
  double lambda;
  int    max_ite;
  UL     seed;
#ifdef OMP 
  int    num_threads;
#endif
};

struct dweak_learner
{
	double *featureIdx;
	double *th;
	double *a;
	double *b;
	double *weaklearner;
	double *epsi;
};

struct sweak_learner
{
	float *featureIdx;
	float *th;
	float *a;
	float *b;
	float *weaklearner;
	float *epsi;
};

/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */
void randini(UL);
void sqs( float * , int , int  ); 
void sqsindex( float * , int * , int , int  );
void stranspose(float*, float * , int , int);
void sgentelboost_decision_stump(float * , float * , struct opts , float * , int ,  float *, float *, float * , float *, int  , int );
void sgentelboost_perceptron(float * , float * , struct opts , float * , int , float *, float *, float * , int  , int );
void dqs( double * , int , int  ); 
void dqsindex( double * , int * , int , int  );
void dtranspose(double *, double * , int , int);
void dgentelboost_decision_stump(double * , double * , struct opts , double * , int ,  double *, double *, double * , double *, int  , int );
void dgentelboost_perceptron(double * , double * , struct opts , double * , int , double *, double *, double * , int  , int );

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{    
    double *dX , *dy; 
	float *sX , *sy;
    int d , N , T=100;
    mxArray *mxtemp;
    struct dweak_learner dmodel;
    struct sweak_learner smodel;
#ifdef OMP 
    struct opts options = {0 , 100 , 1 , 1e-3 , 100 , (UL)NULL , -1};
#else
    struct opts options = {0 , 100 , 1 , 1e-3 , 100 , (UL)NULL};
#endif
    const char *fieldnames_model[6] = {"featureIdx" , "th" , "a" , "b" , "weaklearner" , "epsi"};
    double *tmp , *dysorted , *dlabels;
    float *sysorted , *slabels;
	double temp , dcurrentlabel;
	float scurrentlabel;
    int i , tempint , m=0 , issingle = 0;
	UL templint;

	if (nrhs < 2)
	{
		mexPrintf(
			"\n"
			"\n"
			"\n"
			" Gentle AdaBoost Classifier with two different weak-learners : Decision Stump and Perceptron.\n"
			" Multi-class problem is performed with the one-vs-all strategy.\n"
			"\n"
			"\n"
			" Usage\n"
			" -----\n"
			"\n"
			"\n"
			" model = gentleboost_model(X , y , [options]);\n"
			"\n"
			"\n"
			" Inputs\n"
			" ------\n"
			"\n"
			" X                                Features matrix (d x N) in double precision.\n"
			" y                                Labels vector(1 x N) where y_i={1,...,M} and i=1,...,N.. If y represent binary labels vector then y_i={-1,1}.\n"
			" options\n"
			"        weaklearner               Choice of the weak learner used in the training phase\n"
			"                                  weaklearner = 0 <=> minimizing the weighted error : sum(w * |z - h(x;(th,a,b))|^2) / sum(w), where h(x;(th,a,b)) = (a*(x>th) + b) in R\n"
			"                                  weaklearner = 1 <=> minimizing the weighted error : sum(w * |z - h(x;(a,b))|^2), where h(x;(a,b)) = sigmoid(x ; a,b) in R\n"
			"        T                         Number of weaklearners (default T = 100)\n"
			"        epsi                      Epsilon constant in the sigmoid function used in the perceptron (default epsi = 1.0)\n"
			"        lambda                    Regularization parameter for the perceptron's weights update (default lambda = 1e-3)\n"
			"        max_ite                   Maximum number of iterations of the perceptron algorithm (default max_ite = 100)\n"
			"        seed                      Seed number for internal random generator (default random seed according to time)\n"
#ifdef OMP
			"        num_threads               Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)\n"
#endif			
			"\n"
			"\n"
			"  Output\n"
			"  ------\n"
			"\n"  
			"  model                           Structure of model ouput\n"
			"        featureIdx                Features index in single/double precision of the T best weaklearners (T x m) where m is the number of class.\n" 
			"                                  For binary classification m is force to 1.\n"
			"        th                        Optimal Threshold parameters (1 x T) in single/double precision.\n"
			"        a                         Affine parameter(1 x T) in single/double precision.\n"
			"        b                         Bias parameter (1 x T) in single/double precision.\n"
			"        weaklearner               Choice of the weak learner used in the training phase in single/double precision.\n"
			"        epsi                      Epsilon constant in the sigmoid function used in the perceptron in single/double precision.\n"
			"\n"
			"\n"
			"\n"
			);
		return;		
	}

    /* --------- Input 1 --------------- */
	if(mxIsSingle(prhs[0]))
	{
		sX       = (float *)mxGetData(prhs[0]);
		issingle = 1;
	}
	else
	{
		dX       = (double *)mxGetData(prhs[0]);
	}
	d            = mxGetM(prhs[0]);
	N            = mxGetN(prhs[0]);
    
    /* --------- Input 2 --------------- */

    if ((nrhs > 1) && !mxIsEmpty(prhs[1]) )
    {
		if(issingle)
		{
			sy            = (float *)mxGetPr(prhs[1]);
			sysorted      = (float *)malloc(N*sizeof(float));
			for ( i = 0 ; i < N ; i++ )
			{
				sysorted[i] = sy[i];   
			} 
			sqs( sysorted , 0 , N - 1 );
			slabels       = (float *)malloc(sizeof(float));
			slabels[m]    = sysorted[0];
			scurrentlabel = slabels[0];
			for (i = 0 ; i < N ; i++)
			{
				if (scurrentlabel != sysorted[i])
				{
					slabels       = (float *)realloc(slabels , (m+2)*sizeof(float));
					slabels[++m]  = sysorted[i];
					scurrentlabel = sysorted[i];
				}
			}
			m++;
			if( m == 2) /* Binary case */
			{
				m          = 1;
				slabels[0] = 1.0; /*Force positive label in the first position*/
			}
		}
		else
		{
			dy            = (double *)mxGetPr(prhs[1]);   
			dysorted      = (double *)malloc(N*sizeof(double));
			for ( i = 0 ; i < N ; i++ )
			{
				dysorted[i] = dy[i];   
			} 
			dqs( dysorted , 0 , N - 1 );
			dlabels       = (double *)malloc(sizeof(double));
			dlabels[m]    = dysorted[0];
			dcurrentlabel = dlabels[0];
			for (i = 0 ; i < N ; i++)
			{
				if (dcurrentlabel != dysorted[i])
				{
					dlabels       = (double *)realloc(dlabels , (m+2)*sizeof(double));
					dlabels[++m]  = dysorted[i];
					dcurrentlabel = dysorted[i];
				}
			}
			m++;
			if( m == 2) /* Binary case */
			{
				m         = 1;
				dlabels[0] = 1.0; /*Force positive label in the first position*/
			}
		}
	}
        
    /* Input 3  */
           
    if ((nrhs > 2) && (!mxIsEmpty(prhs[2])) )     
    {
        mxtemp                            = mxGetField(prhs[2] , 0 , "weaklearner");   
        if(mxtemp != NULL)
        {
            tmp                           = mxGetPr(mxtemp);   
            tempint                       = (int) tmp[0];
            if((tempint < 0) || (tempint > 1))
            {
                mexPrintf("weaklearner = {0,1}, force to 0");   
                options.weaklearner        = 0;
            }
            else
            {
                options.weaklearner        = tempint;   
            }
        }

        mxtemp                            = mxGetField(prhs[2] , 0 , "T");   
        if(mxtemp != NULL)
        {
            tmp                           = mxGetPr(mxtemp);   
            tempint                       = (int) tmp[0];
            if(tempint < 1)
            {
                mexPrintf("T > 0, force to 10");   
                options.T                 = 10;
            }
            else
            {
                options.T                 = tempint;   
            }
        }

        mxtemp                            = mxGetField(prhs[2] , 0 , "epsi");
        if(mxtemp != NULL)
        {
			tmp                           = mxGetPr(mxtemp);
			temp                          = tmp[0];
			if(temp < 0.0)
			{
				mexPrintf("T > 0, force to 10");
				options.epsi              = 1.0;
			}
			else
			{
				options.epsi              = temp;
			}
        }
        
        mxtemp                            = mxGetField(prhs[2] , 0 , "lambda");
        if(mxtemp != NULL)
        {
            tmp                           = mxGetPr(mxtemp);
			temp                          = tmp[0];
			if(temp < 0.0)
			{
				mexPrintf("lambda > 0, force to 10e-3");
				options.lambda              = 10e-3; 
			}
			else
			{
				options.lambda            = temp;
			}
        }
        
        mxtemp                            = mxGetField(prhs[2] , 0 , "max_ite");
        if(mxtemp != NULL)
        {
            tmp                           = mxGetPr(mxtemp);
            tempint                       = (int) tmp[0];
            if(tempint < 1)
            {
                mexPrintf("max_ite > 0, force to default value");
                options.max_ite           = 10;
            }
            else
            {
                options.max_ite           =  tempint;
            }
        }

		mxtemp                            = mxGetField(prhs[2] , 0 , "seed");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			templint                      = (UL) tmp[0];
			if( (templint < 1) )
			{
				mexPrintf("seed >= 1 , force to NULL (random seed)\n");	
				options.seed             = (UL)NULL;
			}
			else
			{
				options.seed             = templint;
			}
		}
    }
     
	/*------------------------ Main Call ----------------------------*/

	if(issingle)
	{
		if(options.weaklearner == 0)
		{
			plhs[0]              =  mxCreateStructMatrix(1 , 1 , 6 , fieldnames_model);
			for(i = 0 ; i < 4 ; i++)
			{
				mxSetFieldByNumber(plhs[0] ,0 , i , mxCreateNumericMatrix(options.T , m , mxSINGLE_CLASS , mxREAL));
			}
			mxSetFieldByNumber(plhs[0] , 0 , 4 , mxCreateNumericMatrix(1 , 1 , mxSINGLE_CLASS , mxREAL));
			mxSetFieldByNumber(plhs[0] , 0 , 5 , mxCreateNumericMatrix(1 , 1 , mxSINGLE_CLASS , mxREAL));

			smodel.featureIdx     = (float *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[0] ) );
			smodel.th             = (float *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[1] ) );
			smodel.a              = (float *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[2] ) );
			smodel.b              = (float *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[3] ) );
			smodel.weaklearner    = (float *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[4] ) );
			smodel.epsi           = (float *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[5] ) );

			smodel.weaklearner[0] = (float)options.weaklearner;
			smodel.epsi[0]        = (float)options.epsi;

			sgentelboost_decision_stump(sX , sy , options , slabels , m , smodel.featureIdx , smodel.th , smodel.a , smodel.b , d , N);
		}
		if(options.weaklearner == 1)
		{
			plhs[0]              =  mxCreateStructMatrix(1 , 1 , 6 , fieldnames_model);
			for(i = 0 ; i < 4 ; i++)
			{
				mxSetFieldByNumber(plhs[0] ,0 , i , mxCreateNumericMatrix(options.T , m , mxSINGLE_CLASS,mxREAL));
			}
			mxSetFieldByNumber(plhs[0] ,0 , 4 , mxCreateNumericMatrix(1 , 1 , mxSINGLE_CLASS , mxREAL));
			mxSetFieldByNumber(plhs[0] ,0 , 5 , mxCreateNumericMatrix(1 , 1 , mxSINGLE_CLASS , mxREAL));


			smodel.featureIdx     = (float *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[0] ) );
			smodel.th             = (float *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[1] ) );
			smodel.a              = (float *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[2] ) );
			smodel.b              = (float *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[3] ) );
			smodel.weaklearner    = (float *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[4] ) );
			smodel.epsi           = (float *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[5] ) );

			smodel.weaklearner[0] = (float)options.weaklearner;
			smodel.epsi[0]        = (float)options.epsi;

			randini(options.seed);

			sgentelboost_perceptron(sX , sy  , options , slabels , m , smodel.featureIdx , smodel.a , smodel.b , d , N);
		}
		/* -------------- Free Memory ----------- */

		free(slabels);
		free(sysorted);
	}
	else
	{
		if(options.weaklearner == 0)
		{
			plhs[0]              =  mxCreateStructMatrix(1 , 1 , 6 , fieldnames_model);
			for(i = 0 ; i < 4 ; i++)
			{
				mxSetFieldByNumber(plhs[0] ,0 , i , mxCreateNumericMatrix(options.T , m , mxDOUBLE_CLASS , mxREAL));
			}
			mxSetFieldByNumber(plhs[0] ,0 , 4 , mxCreateNumericMatrix(1 , 1 , mxDOUBLE_CLASS , mxREAL));
			mxSetFieldByNumber(plhs[0] ,0 , 5 , mxCreateNumericMatrix(1 , 1 , mxDOUBLE_CLASS , mxREAL));

			dmodel.featureIdx     = (double *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[0] ) );
			dmodel.th             = (double *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[1] ) );
			dmodel.a              = (double *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[2] ) );
			dmodel.b              = (double *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[3] ) );
			dmodel.weaklearner    = (double *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[4] ) );
			dmodel.epsi           = (double *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[5] ) );

			dmodel.weaklearner[0] = (double)options.weaklearner;
			dmodel.epsi[0]        = options.epsi;

			dgentelboost_decision_stump(dX , dy , options , dlabels , m , dmodel.featureIdx , dmodel.th , dmodel.a , dmodel.b , d , N);
		}
		if(options.weaklearner == 1)
		{
			plhs[0]              =  mxCreateStructMatrix(1 , 1 , 6 , fieldnames_model);
			for(i = 0 ; i < 4 ; i++)
			{
				mxSetFieldByNumber(plhs[0] ,0 , i , mxCreateNumericMatrix(options.T , m , mxDOUBLE_CLASS,mxREAL));
			}
			mxSetFieldByNumber(plhs[0] ,0 , 4 , mxCreateNumericMatrix(1 , 1 , mxDOUBLE_CLASS , mxREAL));
			mxSetFieldByNumber(plhs[0] ,0 , 5 , mxCreateNumericMatrix(1 , 1 , mxDOUBLE_CLASS , mxREAL));


			dmodel.featureIdx     = (double *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[0] ) );
			dmodel.th             = (double *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[1] ) );
			dmodel.a              = (double *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[2] ) );
			dmodel.b              = (double *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[3] ) );
			dmodel.weaklearner    = (double *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[4] ) );
			dmodel.epsi           = (double *)mxGetPr( mxGetField( plhs[0], 0, fieldnames_model[5] ) );

			dmodel.weaklearner[0] = (double)options.weaklearner;
			dmodel.epsi[0]        = options.epsi;

			randini(options.seed);
			dgentelboost_perceptron(dX , dy  , options , dlabels , m , dmodel.featureIdx , dmodel.a , dmodel.b , d , N);
		}

		/* -------------- Free Memory ----------- */

		free(dlabels);
		free(dysorted);
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dgentelboost_decision_stump(double *X , double *y , struct opts options , double *labels , int m, double *featuresIdx, double *th , double *a, double *b, int d , int N)
{ 
    double cteN =1.0/(double)N;
    int i , j , t , c  , cT;
	int T = options.T;
    int indN , Nd = N*d , ind, N1 = N - 1 , featuresIdx_opt , currentlabel , templabel;
    double *w, *Xt, *Xtsorted , *xtemp , *ytemp , *wtemp  ;
    int *Ytsorted , *index, *idX;
    double atemp , btemp  , sumSw , Eyw , fm  , sumwyy , error , errormin, th_opt , a_opt , b_opt , label;
 	double  temp , Sw , Syw;
#ifdef OMP 
    int num_threads = options.num_threads;
    num_threads      = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif
    
	/* Internal allocation */
    
    Xt           = (double *)malloc(Nd*sizeof(double));
    Xtsorted     = (double *)malloc(Nd*sizeof(double));
    Ytsorted     = (int *)malloc(Nd*sizeof(int));
    idX          = (int *)malloc(Nd*sizeof(int));
    w            = (double *)malloc(N*sizeof(double));
    index        = (int *)malloc(N*sizeof(int));

#ifdef OMP 
	xtemp         = (double *)malloc(N*sizeof(double));
#else
    wtemp        = (double *)malloc(N*sizeof(double));
 	ytemp        = (double *)malloc(N*sizeof(double));
    xtemp        = (double *)malloc(N*sizeof(double));
#endif
     
    /* Transpose data to speed up computation */
    
    dtranspose(X , Xt , d , N);
    
    /* Sorting data to speed up computation */
    
    indN   = 0;
    for(j = 0 ; j < d ; j++)
    {
        for(i = 0 ; i < N ; i++)
        {
            index[i] = i;
            xtemp[i] = Xt[i + indN];
        }

		dqsindex(xtemp , index , 0 , N1);
		for(i = 0 ; i < N ; i++)
        {
            ind                = index[i];
            Xtsorted[i + indN] = xtemp[i];
            Ytsorted[i + indN] = (int)y[ind];
            idX[i + indN]      = ind;
        }
        indN        += N;
    }
#ifdef OMP
	free(xtemp);
#endif

    cT   = 0;
    for (c = 0 ; c < m ; c++)
    {
		label        = labels[c];
        currentlabel = (int)label;
        for(i = 0 ; i < N ; i++)
        {
            w[i]     = cteN;
        }
        for(t = 0 + cT ; t < T + cT; t++)
        {
            errormin         = huge;
 /*           indN             = 0; */

#ifdef OMP 
#pragma omp parallel  default(none) private(xtemp,ind,ytemp,wtemp,j,i,atemp,error,Syw,Sw,indN,btemp,templabel,temp) shared(d,N,N1,idX,Xtsorted,w,Ytsorted,Eyw,sumwyy,featuresIdx_opt,th_opt,a_opt,b_opt, errormin,currentlabel)
#endif
			{
#ifdef OMP 
				wtemp               = (double *)malloc(N*sizeof(double));
				ytemp               = (double *)malloc(N*sizeof(double));
				xtemp               = (double *)malloc(N*sizeof(double));
#else
#endif

#ifdef OMP 
#pragma omp for nowait
#endif
				for(j = 0 ; j < d  ; j++)
				{
					indN             = j*N;
					Eyw              = 0.0;
					sumwyy           = 0.0;
					for(i = 0 ; i < N ; i++)
					{
						ind       = i + indN;
						xtemp[i]  = Xtsorted[ind];
						templabel = Ytsorted[ind];
						if(templabel == currentlabel)
						{
							ytemp[i]  = 1.0;
						}
						else
						{
							ytemp[i]  = -1.0;
						}
						wtemp[i]   = w[idX[ind]];
						temp       = ytemp[i]*wtemp[i];
						Eyw       += temp;
						sumwyy    += (ytemp[i]*temp);
					}
					Sw          = 0.0;
					Syw         = 0.0;
					for(i = 0 ; i < N ; i++)
					{
						ind         = i + indN;
						Sw         += wtemp[i];
						Syw        += (ytemp[i]*wtemp[i]);
						btemp       = Syw/Sw;
						if(Sw != 1.0)
						{
							atemp  = (Eyw - Syw)/(1.0 - Sw) - btemp;
						}
						else
						{
							atemp  = (Eyw - Syw) - btemp;
						}

						error   = sumwyy - 2.0*atemp*(Eyw - Syw) - 2.0*btemp*Eyw + (atemp*atemp + 2.0*atemp*btemp)*(1.0 - Sw) + btemp*btemp;
						if(error < errormin)					
						{
							errormin        = error;
							featuresIdx_opt = j;
							if(i < N1)
							{
								th_opt     = (xtemp[i] + xtemp[i + 1])*0.5;
							}
							else
							{
								th_opt     = xtemp[i];
							}
							a_opt          = atemp;
							b_opt          = btemp;					
						}
					}
					/*                indN   += N; */
				}
#ifdef OMP
				free(xtemp);
				free(wtemp);
				free(ytemp);
#else

#endif
			}
            
			/* Best parameters for weak-learner t */
            
            featuresIdx[t]   = (double) (featuresIdx_opt + 1);
            th[t]            = th_opt;
            a[t]             = a_opt;
            b[t]             = b_opt;
            
			/* Weigth's update */
            
            ind              = featuresIdx_opt*N;
            sumSw            = 0.0;
            for (i = 0 ; i < N ; i++)
            {
                fm       = a_opt*(Xt[i + ind] > th_opt) + b_opt;
				if(y[i] == label)
				{
					w[i]    *= exp(-fm);
				}
				else
				{
					w[i]    *= exp(fm);
				}
                sumSw   += w[i];
            }

			sumSw            = 1.0/sumSw;
            for (i = 0 ; i < N ; i++)
            {
                w[i]         *= sumSw;
            }
        }
        cT    += T;
    }
    
    free(w);
    free(Xt);
    free(Xtsorted);
    free(Ytsorted);
    free(index);
    free(idX);
#ifdef OMP

#else
	free(ytemp);
	free(xtemp);
	free(wtemp);
#endif
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void sgentelboost_decision_stump(float *X , float *y , struct opts options , float *labels , int m, float *featuresIdx, float *th , float *a, float *b, int d , int N)
{ 
    float cteN = 1.0f/(float)N;
    int i , j , t , c  , cT;
	int T = options.T;
    int indN , Nd = N*d , ind, N1 = N - 1 , featuresIdx_opt , currentlabel , templabel;
    float *w, *Xt, *Xtsorted , *xtemp , *ytemp , *wtemp  ;
    int *Ytsorted , *index, *idX;
    float atemp , btemp  , sumSw , Eyw , fm  , sumwyy , error , errormin, th_opt , a_opt , b_opt , label;
 	float  temp , Sw , Syw;
#ifdef OMP 
    int num_threads = options.num_threads;
    num_threads      = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif

    
	/* Internal allocation */
    
    Xt           = (float *)malloc(Nd*sizeof(float));
    Xtsorted     = (float *)malloc(Nd*sizeof(float));
    Ytsorted     = (int *)malloc(Nd*sizeof(int));
    idX          = (int *)malloc(Nd*sizeof(int));
    ytemp        = (float *)malloc(N*sizeof(float));
    wtemp        = (float *)malloc(N*sizeof(float));
    w            = (float *)malloc(N*sizeof(float));
    xtemp        = (float *)malloc(N*sizeof(float));
    index        = (int *)malloc(N*sizeof(int));
  
#ifdef OMP 
    xtemp        = (float *)malloc(N*sizeof(float));
#else
    ytemp        = (float *)malloc(N*sizeof(float));
    wtemp        = (float *)malloc(N*sizeof(float));
    xtemp        = (float *)malloc(N*sizeof(float));
#endif


	/* Transpose data to speed up computation */

	stranspose(X , Xt , d , N);

	/* Sorting data to speed up computation */

	indN   = 0;
	for(j = 0 ; j < d ; j++)
	{
		for(i = 0 ; i < N ; i++)
		{
			index[i] = i;
			xtemp[i] = Xt[i + indN];
		}

		sqsindex(xtemp , index , 0 , N1);
		for(i = 0 ; i < N ; i++)
		{
			ind                = index[i];
			Xtsorted[i + indN] = xtemp[i];
			Ytsorted[i + indN] = (int)y[ind];
			idX[i + indN]      = ind;
		}
		indN        += N;
	}
#ifdef OMP 
	free(xtemp);
#endif
	cT   = 0;
	for (c = 0 ; c < m ; c++)
	{
		label        = labels[c];
		currentlabel = (int)label;
		for(i = 0 ; i < N ; i++)
		{
			w[i]     = cteN;
		}
		for(t = 0 + cT ; t < T + cT; t++)
		{
			errormin         = huge;
			/*            indN             = 0; */

#ifdef OMP 
#pragma omp parallel  default(none) private(xtemp,ind,ytemp,wtemp,j,i,atemp,error,Syw,Sw,indN,btemp,templabel,temp) shared(d,N,N1,idX,Xtsorted,w,Ytsorted,Eyw,sumwyy,featuresIdx_opt,th_opt,a_opt,b_opt, errormin,currentlabel)
#endif
			{
#ifdef OMP 
				wtemp               = (float *)malloc(N*sizeof(float));
				ytemp               = (float *)malloc(N*sizeof(float));
				xtemp               = (float *)malloc(N*sizeof(float));
#else
#endif

#ifdef OMP 
#pragma omp for nowait
#endif
				for(j = 0 ; j < d  ; j++)
				{
					indN             = j*N;
					Eyw              = 0.0;
					sumwyy           = 0.0;
					for(i = 0 ; i < N ; i++)
					{
						ind       = i + indN;
						xtemp[i]  = Xtsorted[ind];
						templabel = Ytsorted[ind];
						if(templabel == currentlabel)
						{
							ytemp[i]  = 1.0f;
						}
						else
						{
							ytemp[i]  = -1.0f;
						}
						wtemp[i]   = w[idX[ind]];
						temp       = ytemp[i]*wtemp[i];
						Eyw       += temp;
						sumwyy    += (ytemp[i]*temp);
					}
					Sw          = 0.0f;
					Syw         = 0.0f;
					for(i = 0 ; i < N ; i++)
					{
						ind         = i + indN;
						Sw         += wtemp[i];
						Syw        += (ytemp[i]*wtemp[i]);
						btemp       = Syw/Sw;
						if(Sw != 1.0f)
						{
							atemp  = (Eyw - Syw)/(1.0f - Sw) - btemp;
						}
						else
						{
							atemp  = (Eyw - Syw) - btemp;
						}

						error   = sumwyy - 2.0f*atemp*(Eyw - Syw) - 2.0f*btemp*Eyw + (atemp*atemp + 2.0f*atemp*btemp)*(1.0f - Sw) + btemp*btemp;
						if(error < errormin)					
						{
							errormin        = error;
							featuresIdx_opt = j;
							if(i < N1)
							{
								th_opt     = (xtemp[i] + xtemp[i + 1])*0.5f;
							}
							else
							{
								th_opt     = xtemp[i];
							}
							a_opt          = atemp;
							b_opt          = btemp;					
						}
					}
					/*                indN   += N; */
				}
#ifdef OMP
				free(xtemp);
				free(wtemp);
				free(ytemp);
#else

#endif
			}
            
			/* Best parameters for weak-learner t */
            
            featuresIdx[t]   = (float) (featuresIdx_opt + 1);
            th[t]            = th_opt;
            a[t]             = a_opt;
            b[t]             = b_opt;
            
			/* Weigth's update */
            
            ind              = featuresIdx_opt*N;
            sumSw            = 0.0f;
            for (i = 0 ; i < N ; i++)
            {
                fm       = a_opt*(Xt[i + ind] > th_opt) + b_opt;
				if(y[i] == label)
				{
					w[i]    *= (float)exp(-fm);
				}
				else
				{
					w[i]    *= (float)exp(fm);
				}
                sumSw   += w[i];
            }

			sumSw            = 1.0f/sumSw;
            for (i = 0 ; i < N ; i++)
            {
                w[i]         *= sumSw;
            }
        }
        cT    += T;
    }
    
    free(w);
    free(Xt);
    free(Xtsorted);
    free(Ytsorted);
    free(index);
    free(idX);
#ifdef OMP

#else
	free(ytemp);
	free(xtemp);
	free(wtemp);
#endif

}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  dgentelboost_perceptron(double *X , double *y , struct opts options , double *labels , int m , double *featuresIdx, double *a , double *b, int d , int N)
{
    double epsi = options.epsi , lambda = options.lambda , error , errormin , cteN =1.0/(double)N;
    int t , j , i , k , c , cT ;
	int T = options.T;
    int featuresIdx_opt;
    int max_ite = options.max_ite, Nd = N*d , indN , index;
    double *Xt , *w;
    double *ytemp;
    double atemp , btemp , xi  , temp , fx , tempyifx , sum , fm , currentlabel;
    double a_opt, b_opt;
     
	/* Internal allocation */

    Xt           = (double *)malloc(Nd*sizeof(double));
    w            = (double *)malloc(N*sizeof(double));
    ytemp        = (double *)malloc(N*sizeof(double));
    
	/* Transpose data to speed up computation */

	dtranspose(X , Xt , d , N);
    
    cT  = 0;  
    for (c = 0 ; c < m ; c++)
    {  
        currentlabel = labels[c];
        for(i = 0 ; i < N ; i++)
        {
            w[i]    = cteN;
            if(y[i] == currentlabel)
            {
				ytemp[i]  = 1.0;
            }
            else
            {
                ytemp[i]  = -1.0;
            }
        }
        
        for(t = 0 + cT ; t < T + cT ; t++)
        {
            errormin = huge;
            indN     = 0;

			for(j = 0 ; j < d  ; j++)
            {
                /* Random initialisation of weights */
                index  = (int)floor(N*rand());
                atemp  = Xt[index + indN];
                index  = (int)floor(N*rand());
                btemp  = Xt[index + indN];
                
                /* Weight's optimization  */
                for(k = 0 ; k < max_ite ; k++)
                {             
					for(i = 0 ; i < N ; i++)
                    {
                        xi         = Xt[i + indN];
                        fx         = (2.0/( 1.0 + exp(-2.0*epsi*(atemp*xi + btemp)) )) - 1.0; /* sigmoid in [-1 , 1] */
                        temp       = lambda*(ytemp[i] - fx)*epsi*(1.0 - fx*fx);	/* d(sig(x;epsi))/dx = epsi*(1 - fx²) */
                        atemp     += (temp*xi);
                        btemp     += temp;
                    }
                }
                
                /* Weigthed error */

				error         = 0.0;
                for(i = 0 ; i < N ; i++)
                {
                    fx        = (2.0/(1.0 + exp(-2.0*epsi*(atemp*Xt[i + indN] + btemp)))) - 1.0;
                    tempyifx  = (ytemp[i] - fx);
                    error    += w[i]*tempyifx*tempyifx;
                }
                if(error < errormin)
                {
                    errormin        = error;
                    featuresIdx_opt = j;
                    a_opt           = atemp;
                    b_opt           = btemp;
                }
                indN    += N;
            }
            featuresIdx[t]   = (double) (featuresIdx_opt + 1);
            a[t]             = a_opt;
            b[t]             = b_opt;
            
            index            = featuresIdx_opt*N;
            sum              = 0.0;
            for (i = 0 ; i < N ; i++)
            {
			    fm           = (2.0/(1.0 + exp(-2.0*epsi*(a_opt*Xt[i + index] + b_opt)))) - 1.0;
                w[i]        *= exp(-ytemp[i]*fm);
                sum         += w[i];
            }
            
            sum              = 1.0/sum;
            for (i = 0 ; i < N ; i++)
            {
                w[i]         *= sum;
            }
        }
        cT    += T;      
    }
    free(Xt);
    free(w);
    free(ytemp);
}


/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  sgentelboost_perceptron(float *X , float *y , struct opts options , float *labels , int m , float *featuresIdx, float *a , float *b, int d , int N)
{
    float epsi = (float)options.epsi , lambda = (float)options.lambda , error , errormin , cteN =1.0f/(float)N;
    int t , j , i , k , c , cT ;
	int T = options.T;
    int featuresIdx_opt;
    int max_ite = options.max_ite, Nd = N*d , indN , index;
    float *Xt , *w;
    float *ytemp;
    float atemp , btemp , xi  , temp , fx , tempyifx , sum , fm , currentlabel;
    float a_opt, b_opt;
     
	/* Internal allocation */

    Xt           = (float *)malloc(Nd*sizeof(float));
    w            = (float *)malloc(N*sizeof(float));
    ytemp        = (float *)malloc(N*sizeof(float));
    
	/* Transpose data to speed up computation */

	stranspose(X , Xt , d , N);
    
    cT  = 0;  
    for (c = 0 ; c < m ; c++)
    {  
        currentlabel = labels[c];
        for(i = 0 ; i < N ; i++)
        {
            w[i]    = cteN;
            if(y[i] == currentlabel)
            {
				ytemp[i]  = 1.0f;
            }
            else
            {
                ytemp[i]  = -1.0f;
            }
        }
        
        for(t = 0 + cT ; t < T + cT ; t++)
        {
            errormin = huge;
            indN     = 0;

			for(j = 0 ; j < d  ; j++)
            {
                /* Random initialisation of weights */
                index  = (int)floor(N*rand());
                atemp  = Xt[index + indN];
                index  = (int)floor(N*rand());
                btemp  = Xt[index + indN];
                
                /* Weight's optimization  */
                for(k = 0 ; k < max_ite ; k++)
                {             
					for(i = 0 ; i < N ; i++)
                    {
                        xi         = Xt[i + indN];
                        fx         = (2.0f/( 1.0f + (float)exp(-2.0f*epsi*(atemp*xi + btemp)) )) - 1.0f; /* sigmoid in [-1 , 1] */
                        temp       = lambda*(ytemp[i] - fx)*epsi*(1.0f - fx*fx);	/* d(sig(x;epsi))/dx = epsi*(1 - fx²) */
                        atemp     += (temp*xi);
                        btemp     += temp;
                    }
                }
                
                /* Weigthed error */

				error         = 0.0f;
                for(i = 0 ; i < N ; i++)
                {
                    fx        = (2.0f/(1.0f + (float)exp(-2.0f*epsi*(atemp*Xt[i + indN] + btemp)))) - 1.0f;
                    tempyifx  = (ytemp[i] - fx);
                    error    += (w[i]*tempyifx*tempyifx);
                }
                if(error < errormin)
                {
                    errormin        = error;
                    featuresIdx_opt = j;
                    a_opt           = atemp;
                    b_opt           = btemp;
                }
                indN    += N;
            }
            featuresIdx[t]   = (float) (featuresIdx_opt + 1);
            a[t]             = a_opt;
            b[t]             = b_opt;
            
            index            = featuresIdx_opt*N;
            sum              = 0.0f;
            for (i = 0 ; i < N ; i++)
            {
			    fm           = (2.0f/(1.0f + (float)exp(-2.0f*epsi*(a_opt*Xt[i + index] + b_opt)))) - 1.0f;
                w[i]        *= (float)exp(-ytemp[i]*fm);
                sum         += w[i];
            }
            
            sum              = 1.0f/sum;
            for (i = 0 ; i < N ; i++)
            {
                w[i]         *= sum;
            }
        }
        cT    += T;      
    }
    free(Xt);
    free(w);
    free(ytemp);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dqs(double  *a , int lo, int hi)
{
/*  lo is the lower index, hi is the upper index
    of the region of array a that is to be sorted
*/
    int i=lo, j=hi;
    double x=a[(lo+hi)/2] , h;

    /*  partition  */
    do
    {    
        while (a[i]<x) i++; 
        while (a[j]>x) j--;
        if (i<=j)
        {
            h        = a[i]; 
			a[i]     = a[j]; 
			a[j]     = h;
            i++; 
			j--;
        }
    }
	while (i<=j);

    /*  recursion  */

    if (lo<j) dqs(a , lo , j);
    if (i<hi) dqs(a , i , hi);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dqsindex (double *a, int *index , int lo, int hi)
{
/*  lo is the lower index, hi is the upper index
    of the region of array a that is to be sorted
*/
    int i=lo, j=hi , ind;
    double x=a[(lo+hi)/2] , h;

    /* partition */
    do
    {    
        while (a[i]<x) i++; 
        while (a[j]>x) j--;
        if (i<=j)
        {
            h        = a[i]; 
			a[i]     = a[j]; 
			a[j]     = h;
			ind      = index[i];
			index[i] = index[j];
			index[j] = ind;
            i++; 
			j--;
        }
    }
	while (i<=j);

    /*  recursion */
    if (lo<j) dqsindex(a , index , lo , j);
    if (i<hi) dqsindex(a , index , i , hi);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dtranspose(double *A, double *B , int m , int n)
{
    int i , j , jm = 0, in;
        
    for (j = 0 ; j<n ; j++)    
    {
        in  = 0;
        for(i = 0 ; i<m ; i++)
        {
            B[j + in] = A[i + jm];
            in       += n;
        }
        jm  += m;
    }
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void sqs(float *a , int lo, int hi)
{
/*  lo is the lower index, hi is the upper index
    of the region of array a that is to be sorted
*/
    int i=lo, j=hi;
    float x=a[(lo+hi)/2] , h;

    /*  partition  */
    do
    {    
        while (a[i]<x) i++; 
        while (a[j]>x) j--;
        if (i<=j)
        {
            h        = a[i]; 
			a[i]     = a[j]; 
			a[j]     = h;
            i++; 
			j--;
        }
    }
	while (i<=j);

    /*  recursion  */

    if (lo<j) sqs(a , lo , j);
    if (i<hi) sqs(a , i , hi);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void sqsindex (float  *a, int *index , int lo, int hi)
{
/*  lo is the lower index, hi is the upper index
    of the region of array a that is to be sorted
*/
    int i=lo, j=hi , ind;
    float x=a[(lo+hi)/2] , h;

    /* partition */
    do
    {    
        while (a[i]<x) i++; 
        while (a[j]>x) j--;
        if (i<=j)
        {
            h        = a[i]; 
			a[i]     = a[j]; 
			a[j]     = h;
			ind      = index[i];
			index[i] = index[j];
			index[j] = ind;
            i++; 
			j--;
        }
    }
	while (i<=j);

    /*  recursion */
    if (lo<j) sqsindex(a , index , lo , j);
    if (i<hi) sqsindex(a , index , i , hi);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void stranspose(float *A, float *B , int m , int n)
{
    int i , j , jm = 0, in;
        
    for (j = 0 ; j<n ; j++)    
    {
        in  = 0;
        for(i = 0 ; i<m ; i++)
        {
            B[j + in] = A[i + jm];
            in       += n;
        }
        jm  += m;
    }
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void randini(UL seed)
{
	/* SHR3 Seed initialization */

	if(seed == (UL)NULL)
	{
		jsrseed  = (UL) time( NULL );
		jsr     ^= jsrseed;
	}
	else
	{
		jsr     = (UL)NULL;
		jsrseed = seed;
		jsr    ^= jsrseed;
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
