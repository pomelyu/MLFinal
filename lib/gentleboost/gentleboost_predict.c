/*

  Predict data for a Strong Classifier trained with Gentle AdoBoost classifier

  Usage
  -----

  [yest , fx] = gentleboost_predict(X , model);

  
  Inputs
  ------

  X                                     Features matrix (d x N) in single/double precision

  model                                 Structure of model ouput
	          featureIdx                Features index in single/double precision of the T best weaklearners (T x m) where m is the number of class. 
			                            For binary classification m is force to 1.
			  th                        Optimal Threshold parameters (1 x T) in single/double precision.
			  a                         Affine parameter(1 x T) in single/double precision.
			  b                         Bias parameter (1 x T) in single/double precision.
			  weaklearner               Choice of the weak learner used in the training phase.
			  epsi                      Epsilon constant in the sigmoid function used in the perceptron.
  
  Outputs
  -------
  
  yest                                  Estimated labels (1 x N), yest_i = {1,...,m} in single/double precision.
  fx                                    Output of the strong classifier (1 x N) in single/double precision.

  To compile
  ----------

  mex  gentleboost_predict.c

  mex  -output gentleboost_predict.dll gentleboost_predict.c

  mex  -f mexopts_intel10.bat -output gentleboost_predict.dll gentleboost_predict.c


  Example 1
  ---------

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


 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 01/27/2009

 Reference  : [1] R.E Schapire and al "Boosting the margin : A new explanation for the effectiveness of voting methods". 
 ---------        The annals of statistics, 1999


 Changelog  
 --------- 
            v1.1 Minor updates (11/24/2011)
            v1.0 Initial release (01/27/2009)
*/


#include <math.h>
#include <mex.h>

#define sign(a) ((a) >= (0) ? (1.0) : (-1.0))
#define tiny -1e300

struct opts
{
  int   weaklearner;
  double epsi;
};

struct dweak_learner
{
	double *featureIdx;
	double *th;
	double *a;
	double *b;
};

struct sweak_learner
{
	float *featureIdx;
	float *th;
	float *a;
	float *b;
};


/*-------------------------------------------------------------------------------------------------------------- */
/* Function prototypes */

void  sgentleboost_predict(float * , struct sweak_learner  , struct opts  , float * , float *, int  , int, int , int );
void  dgentleboost_predict(double * , struct dweak_learner  , struct opts  , double * , double *, int  , int, int , int );

/*-------------------------------------------------------------------------------------------------------------- */
void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray *prhs[] )
{
    double *dX;
	float *sX;
	struct dweak_learner dmodel;
	struct sweak_learner smodel;
	struct opts options = {0 , 1.0};
	double *dyest , *dfx;
	float *syest , *sfx;
	double temp;
	int d , N , T = 100 , m = 1 , issingle = 0;
	mxArray *mxtemp;
	double *tmp;
	int tempint;

	if (nrhs < 1)
	{
		mexPrintf(
			"\n"
			"\n"
			" Predict data for a Strong Classifier trained with Gentle AdoBoost classifier.\n"
			"\n"
			"\n"
			" Usage\n"
			" -----\n"
			"\n"
			"\n"
			" [yest , fx] = gentleboost_predict(X , model);\n"
			"\n"
			"\n"  
			" Inputs\n"
			" ------\n"
			"\n"
			"\n"
			" X                             Features matrix (d x N) in single/double precision\n"
			" model                         Structure of model ouput:\n"
			"      featureIdx               Features index in single/double precision of the T best weaklearners (T x m) where m is the number of class.\n" 
			"                               For binary classification m is force to 1.\n"
			"      th                       Optimal Threshold parameters (1 x T) in single/double precision.\n"
			"      a                        Affine parameter(1 x T) in single/double precision.\n"
			"      b                        Bias parameter (1 x T) in single/double precision.\n"
			"      weaklearner              Choice of the weak learner used in the training phase.\n"
			"      epsi                     Epsilon constant in the sigmoid function used in the perceptron.\n"
			"\n"
			"\n"
			" Outputs\n"
			" -------\n"
			"\n"
			"\n"
			" yest                          Estimated labels (1 x N), yest_i = {1,...,m} in single/double precision.\n"
			" fx                            Output of the strong classifier (1 x N) in single/double precision.\n"
			"\n"
			"\n"
			);
			return;
	}
	/* ---------- Input 1  ----------------- */
	
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
	
	/* ---------- Input 2  ----------------- */
	
	if (!mxIsEmpty(prhs[1]) && mxIsStruct(prhs[1]) )	
	{
		if(issingle)
		{
			mxtemp                            = mxGetField( prhs[1], 0, "featureIdx" );	
			if(mxtemp != NULL)
			{
				smodel.featureIdx             = (float *)mxGetPr(mxtemp);
				T                             = mxGetM(mxtemp);
				m                             = mxGetN(mxtemp);
			}

			mxtemp                            = mxGetField( prhs[1], 0, "th" );
			if(mxtemp != NULL)
			{
				smodel.th                     = (float *)mxGetPr(mxtemp);	  
			}

			mxtemp                            = mxGetField( prhs[1] , 0, "a" );
			if(mxtemp != NULL)
			{
				smodel.a                      = (float *)mxGetPr(mxtemp);	
			}

			mxtemp                            = mxGetField( prhs[1] , 0, "b" );
			if(mxtemp != NULL)
			{
				smodel.b                      = (float *)mxGetPr(mxtemp);	
			}
		}
		else
		{
			mxtemp                            = mxGetField( prhs[1], 0, "featureIdx" );	
			if(mxtemp != NULL)
			{
				dmodel.featureIdx             = mxGetPr(mxtemp);
				T                             = mxGetM(mxtemp);
				m                             = mxGetN(mxtemp);
			}

			mxtemp                            = mxGetField( prhs[1], 0, "th" );
			if(mxtemp != NULL)
			{
				dmodel.th                     = mxGetPr(mxtemp);	  
			}

			mxtemp                            = mxGetField( prhs[1] , 0, "a" );
			if(mxtemp != NULL)
			{
				dmodel.a                      = mxGetPr(mxtemp);	
			}

			mxtemp                            = mxGetField( prhs[1] , 0, "b" );
			if(mxtemp != NULL)
			{
				dmodel.b                      = mxGetPr(mxtemp);	
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "weaklearner");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);	
			tempint                       = (int) tmp[0];
			if(tempint < 0)
			{
				mexPrintf("weaklearner ={0,1}, force default to 0");		
				options.weaklearner       = 0;
			}
			else
			{
				options.weaklearner       = tempint;	
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "epsi");
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
	}
	else
	{	
		mexErrMsgTxt("model must be a structure");	
	}
		
		/*------------------------ Main Call ----------------------------*/

	if(issingle)
	{
		/* ---------- Output 1  ----------------- */

		plhs[0]                               =  mxCreateNumericMatrix(1 , N , mxSINGLE_CLASS, mxREAL);
		syest                                 =  (float *)mxGetPr(plhs[0]);

		/* ---------- Output 2  ----------------- */

		plhs[1]                               =  mxCreateNumericMatrix(m , N, mxSINGLE_CLASS, mxREAL);
		sfx                                   =  (float *) mxGetPr(plhs[1]);

		sgentleboost_predict(sX , smodel , options , syest , sfx , d , N , T , m);	

	}
	else
	{
		/* ---------- Output 1  ----------------- */

		plhs[0]                               =  mxCreateNumericMatrix(1 , N, mxDOUBLE_CLASS, mxREAL);
		dyest                                 =  mxGetPr(plhs[0]);

		/* ---------- Output 2  ----------------- */

		plhs[1]                               =  mxCreateNumericMatrix(m , N, mxDOUBLE_CLASS, mxREAL);
		dfx                                   =  mxGetPr(plhs[1]);

		dgentleboost_predict(dX , dmodel , options , dyest , dfx , d , N , T , m);	

	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  dgentleboost_predict(double *X , struct dweak_learner model , struct opts options , double *yest, double *fx, int d , int N , int T , int m)
{
	int t , n , c , cT , nm;
	int indd , ind_maxi;
	double *featureIdx , *th , *a , *b;
	int weaklearner = (int)options.weaklearner;
	double epsi = options.epsi;
	double sum , maxi;

	if(weaklearner == 0) /* Decision Stump */
	{
		featureIdx = model.featureIdx;
		th         = model.th;	
		a          = model.a;
		b          = model.b;		
		if(m == 1)
		{
			indd      = 0;
			
			for(n = 0 ; n < N ; n++)
			{	
				sum   = 0.0;	
				for(t = 0 ; t < T ; t++)
				{
					sum    += ( a[t]*( X[(int)featureIdx[t] - 1 + indd]>th[t] ) + b[t] );	
				}
				fx[n]       = sum;
				yest[n]     = sign(sum);
				indd       += d;
			}	
		}
		else
		{	
			indd      = 0;	
			nm        = 0;
			for(n = 0 ; n < N ; n++)
			{
				cT        = 0;	
				maxi      = tiny;
				ind_maxi  = 0;
				for(c = 0 ; c < m ; c++)
				{
					sum   = 0.0;
					for(t = 0 + cT ; t < T + cT ; t++)
					{
						sum    += (a[t]*( X[(int)featureIdx[t] - 1 + indd] > th[t] ) + b[t] );	
					}
					fx[c + nm] = sum;
					if(sum > maxi)
					{
						maxi     = sum;	
						ind_maxi = c;
					}		
					cT         += T;
				}
				yest[n]       = (double)ind_maxi;
				nm           += m;
				indd         += d;
			}			
		}
	}
	
	if(weaklearner == 1) /* Perceptron */
	{
		featureIdx = model.featureIdx;		
		a          = model.a;
		b          = model.b;	
		if(m == 1)
		{
			indd       = 0;	
			for(n = 0 ; n < N ; n++)
			{
				sum   = 0.0;	
				for(t = 0 ; t < T ; t++)
				{
					sum    += ((2.0/(1.0 + exp(-2.0*epsi*(a[t]* X[(int)featureIdx[t] - 1 + indd] + b[t])))) - 1.0);	
				}
				fx[n]       = sum;
				yest[n]     = sign(sum);
				indd       += d;
			}
		}
		else
		{	
			indd       = 0;	
			nm         = 0;
			for(n = 0 ; n < N ; n++)
			{
				cT        = 0;		
				maxi      = tiny;
				ind_maxi  = 0;
				for(c = 0 ; c < m ; c++)
				{
					sum        = 0.0;	
					for(t = 0 + cT ; t < T + cT ; t++)
					{
						sum    += ((2.0/(1.0 + exp(-2.0*epsi*(a[t]*X[(int)featureIdx[t] - 1 + indd] + b[t] )))) - 1.0);	
					}
					fx[c + nm] = sum;
					if(sum > maxi)
					{
						maxi     = sum;	
						ind_maxi = c;
					}
					cT         += T;
				}
				yest[n]       = (double)ind_maxi;
				nm           += m;
				indd         += d;
			}
		}	
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void  sgentleboost_predict(float *X , struct sweak_learner model , struct opts options , float *yest, float *fx, int d , int N , int T , int m)
{
	int t , n , c , cT , nm;
	int indd , ind_maxi;
	float *featureIdx , *th , *a , *b;
	int weaklearner = (int) options.weaklearner;
	float epsi = (float) options.epsi;
	float sum , maxi;

	if(weaklearner == 0) /* Decision Stump */
	{
		featureIdx = model.featureIdx;
		th         = model.th;	
		a          = model.a;
		b          = model.b;		
		if(m == 1)
		{
			indd      = 0;
			
			for(n = 0 ; n < N ; n++)
			{	
				sum   = 0.0;	
				for(t = 0 ; t < T ; t++)
				{
					sum    += ( a[t]*( X[(int)featureIdx[t] - 1 + indd]>th[t] ) + b[t] );	
				}
				fx[n]       = sum;
				yest[n]     = (float)sign(sum);
				indd       += d;
			}	
		}
		else
		{	
			indd      = 0;	
			nm        = 0;
			for(n = 0 ; n < N ; n++)
			{
				cT        = 0;	
				maxi      = tiny;
				ind_maxi  = 0;
				for(c = 0 ; c < m ; c++)
				{
					sum   = 0.0;
					for(t = 0 + cT ; t < T + cT ; t++)
					{
						sum    += (a[t]*( X[(int)featureIdx[t] - 1 + indd] > th[t] ) + b[t] );	
					}
					fx[c + nm] = sum;
					if(sum > maxi)
					{
						maxi     = sum;	
						ind_maxi = c;
					}		
					cT         += T;
				}
				yest[n]       = (float)ind_maxi;
				nm           += m;
				indd         += d;
			}			
		}
	}
	
	if(weaklearner == 1) /* Perceptron */
	{
		featureIdx = model.featureIdx;		
		a          = model.a;
		b          = model.b;	
		if(m == 1)
		{
			indd       = 0;	
			for(n = 0 ; n < N ; n++)
			{
				sum   = 0.0;	
				for(t = 0 ; t < T ; t++)
				{
					sum    += ((2.0f/(1.0f + (float)exp(-2.0f*epsi*(a[t]* X[(int)featureIdx[t] - 1 + indd] + b[t])))) - 1.0f);	
				}
				fx[n]       = sum;
				yest[n]     = sign(sum);
				indd       += d;
			}
		}
		else
		{	
			indd       = 0;	
			nm         = 0;
			for(n = 0 ; n < N ; n++)
			{
				cT        = 0;		
				maxi      = tiny;
				ind_maxi  = 0;
				for(c = 0 ; c < m ; c++)
				{
					sum        = 0.0;	
					for(t = 0 + cT ; t < T + cT ; t++)
					{
						sum    += ((2.0f/(1.0f + (float)exp(-2.0f*epsi*(a[t]*X[(int)featureIdx[t] - 1 + indd] + b[t] )))) - 1.0f);	
					}
					fx[c + nm] = sum;
					if(sum > maxi)
					{
						maxi     = sum;	
						ind_maxi = c;
					}
					cT         += T;
				}
				yest[n]       = (float)ind_maxi;
				nm           += m;
				indd         += d;
			}
		}	
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
