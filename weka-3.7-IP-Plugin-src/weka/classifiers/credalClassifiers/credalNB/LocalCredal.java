package weka.classifiers.credalClassifiers.credalNB;

import java.util.ArrayList;
import java.util.Arrays;



public class LocalCredal extends LocalNaive {

      /** for serialization */
      static final long serialVersionUID = -1478242251770381214L;

	/**Builds feature and output class, and computes the relevant counts for MAR and NON-MAR features
	 */



	public LocalCredal(int numExp, int nClasses, boolean weightedPar){
		super("overlap",weightedPar);
		numClasses=nClasses;
		confMatrix=new int[numClasses][numClasses];
		singleAcc= new double[numExp];
		setAcc= new double[numExp];
		determ= new double[numExp];
		outputSz= new double[numExp];
		lnbclnccD= new double[numExp];
		lnbclnccI= new double[numExp];
		discountedAcc=new double[numExp];
		adaptive=true;
		maxTrials=50;
	}

	/**
	 * Setup non-static vars
	 */
	public void setup(ArrayList< int[]> testingSet){

		classFreqs=new ArrayList<double[]>(2);
		contTables=new ArrayList<double[][][]>(2);
		classFreqs.add(new double[numClasses]);
		contTables.add(new double[numFeats][][]);

		int i,j;
		for (i=0;i<numFeats;i++){
			contTables.get(bwdthIdx)[i]=new double[numValues[i]][];
			for (j=0;j<numValues[i];j++){
				contTables.get(bwdthIdx)[i][j]= new double[numClasses];
			}
		}			

		predictions = new int[testingSet.size()][];
		testInstances=testingSet.size();
		idxComparator = new IndexComparator();
	}
	
	
	/*To be called from WEKA only. It calls the setupStaticVars of the super class and the 
	 * performs all the operations of the setup method of this class, but without 
	 * allocating the data member testInstances (unavailable when calling from WEKA).
	 * Addtionally, it:
	 * - updates currentExp
	 * -set currentInstance to -1
	 * Both taks are usually done by lnbc (and then seen by lncc through static variables, but when calling
	 * from WEKA lnbc will be not available)	
	 */
	public void setupFromWeka(ArrayList<int[]> trSet, Integer[] infoArray){
		super.setupStaticVars(trSet, infoArray);
		//local setup method of LNCC, without using the testInstances data member.
		classFreqs=new ArrayList<double[]>(2);
		contTables=new ArrayList<double[][][]>(2);
		classFreqs.add(new double[numClasses]);
		contTables.add(new double[numFeats][][]);

		int i,j;
		for (i=0;i<numFeats;i++){
			contTables.get(bwdthIdx)[i]=new double[numValues[i]][];
			for (j=0;j<numValues[i];j++){
				contTables.get(bwdthIdx)[i][j]= new double[numClasses];
			}
		}			
		//when calling from WEKA, we do not need this two rows.
		//predictions = new int[testingSet.size()][];
		//testInstances=testingSet.size();
		
		idxComparator = new IndexComparator();
		currentExp++;
		instanceIdx=-1;
	}
	

	/**Classify current instance, tuning adaptively the bandwidth if adaptive==true;
	 * updates the perf. indicators and sorts the instances according to their index (prepared
	 * for the next instance to be classified.) 
	 */
	public void classifyCurrentInst(){
		fillContTable();
		predictions[instanceIdx]=computeClassification(currentInst);
		int counter=1;

		if (adaptive){
			while (predictions[instanceIdx].length>1 && counter<maxTrials){
				updateStartStop();
				
				//==NEW CODE
				if (distWeighted){
					resetTables();
					ranker.computeWeights();
					updateTablesW();
				}
				//===
				else{
				updateTablesUnw();
				}
				predictions[instanceIdx]=computeClassification(currentInst);
				counter++;
			}
		}
		
//==BANDWIDTH DUMPING CURRENTLY COMMENTED OUT
//		try{
//			BufferedWriter out=new BufferedWriter(new FileWriter("/home/giorgio/tmp/bandwidth.csv",true));
//			Integer tmp= stop_inst;
//			out.write(tmp.toString()); 
//			out.newLine();
//			out.close();
//		}
//		catch (IOException ioexc) {
//			System.out.println("Unexpected exception writing bandwidth");
//			System.exit(0);
//		}

		updatePerfStats(currentInst[numFeats]);
		if (instanceIdx==LocalNaive.testInstances-1){
			computePerfIndicators();}

	}
	
	
	/**Classify current instance, tuning adaptively the bandwidth if adaptive==true;
	 * updates the perf. indicators and sorts the instances according to their index (prepared
	 * for the next instance to be classified.) 
	 * TO BE USED FROM WEKA ONLY
	 */
	public int[] classifyInstance(int[] instance){
		int[] prediction;
                //currentInst=instance;
                setCurrentInst(instance);
		fillContTable();
		prediction=computeClassification(currentInst);
		int counter=1;

		if (adaptive){
			while (prediction.length>1 && counter<maxTrials){
				updateStartStop();
				
				//==NEW CODE
				if (distWeighted){
					resetTables();
					ranker.computeWeights();
					updateTablesW();
				}
				//===
				else{
				updateTablesUnw();
				}
				prediction=computeClassification(currentInst);
				counter++;
			}
		}

//		updatePerfStats(currentInst[numFeats]);
//		if (instanceIdx==LocalNaive.testInstances-1){
//			computePerfIndicators();}
		return prediction;

	}





	/**increases the bandwidth tmp_k of the quantity stepsize; the bandwidth is furthermore extended to include
	 * all  the instances (if any) which are at a distance equal to the instance in position tmp_k+stepsize. */
	private void updateStartStop(){
		LocalNaive.start_inst=LocalNaive.stop_inst;

		if (stop_inst+stepsize<trSet.size()){
			LocalNaive.stop_inst+=stepsize;

			while ( (stop_inst<orderIndexes.length-1) &&
					(orderIndexes[stop_inst-1].getDistance().compareTo(orderIndexes[stop_inst].getDistance())==0)) 
			{
				stop_inst++;
			}
		}
		else stop_inst=trSet.size()-1;
	}

	//	private void addCounts(){
	//
	//		int i,j;
	//		int[] currentInstance;
	//		int currentClass;
	//		for (i=0;i<stop_inst;i++){
	//			currentInstance=trSet.get(orderIndexes[i].getIndex());
	//			currentClass=currentInstance[numFeats];
	//			classFreqs.get(bwdthIdx)[currentClass]++;
	//			for (j=0;j<numFeats;j++){
	//				if (currentInstance[j] !=-9999){
	//					contTables.get(bwdthIdx)[j][currentInstance[j]][currentClass]++;
	//				}
	//			}
	//		}
	//	}



	/**
	 * After having classified a single instance, stores temporary information about
	 * whether it was accurate, determinate etc.
	 */	private void updatePerfStats(int realClass){

		 //update vector of indicators
		 if (predictions[instanceIdx].length==1){
			 confMatrix[realClass][predictions[instanceIdx][0]]++;
			 determ[currentExp]++;
			 if (predictions[instanceIdx][0]==realClass){
				 singleAcc[currentExp]++;
			 }
		 }
		 //else, we have an indeterminate classification
		 else{
			 if (Arrays.binarySearch(predictions[instanceIdx],realClass)>= 0){
				 setAcc[currentExp]++;
			 }
			 outputSz[currentExp] += predictions[instanceIdx].length;
		 }


		 //===DEBUG
		 //		 try{
		 //			 BufferedWriter out= new BufferedWriter(new FileWriter("/home/giorgio/tmp/wave/statTest2.csv",true));
		 //			 int left=classFreqs.get(0).length-predictions[instanceIdx].length;
		 //			 int jj;
		 //			 out.newLine();
		 //			 for (jj=0;jj<predictions[instanceIdx].length;jj++){
		 //				 out.write(predictions[instanceIdx][jj]+",");
		 //			 }
		 //			 for (jj=0;jj<left;jj++){
		 //				 out.write("-,");
		 //			 }
		 //			 if (Arrays.binarySearch(predictions[instanceIdx],realClass)>= 0){
		 //			 out.write((double)1/predictions[instanceIdx].length+",");
		 //			 }
		 //			 else{
		 //				 out.write("0,");
		 //			 }
		 //			 out.close();
		 //
		 //			 //eventually, update multiLabel Acc
		 //			 if (Arrays.binarySearch(predictions[instanceIdx],realClass)>= 0){
		 //				 multiLabAcc[currentExp] += (double)1/predictions[instanceIdx].length;
		 //			 }
		 //		 }
		 //
		 //		 catch (IOException ioexc) {
		 //			 System.out.println("Unexpected exception writing test to file");
		 //			 System.exit(0);
		 //		 } 
		 //=====END DEBUG



		 //eventually, update multiLabel Acc
		 if (Arrays.binarySearch(predictions[instanceIdx],realClass)>= 0){
			 discountedAcc[currentExp] += (double)1/predictions[instanceIdx].length;
		 }

	 }
	 /**
	  * At the and of a single training/testing experiment, computes
	  *the actual indicators starting from the raw information which is stored in
	  *different arrays. Fills then the ultimate values of such indicators into the
	  *arrays.
	  */
	 private void computePerfIndicators(){

		 if (determ[currentExp]>0){
			 singleAcc[currentExp]/=determ[currentExp];
			 lnbclnccD[currentExp]/=determ[currentExp];
		 }
		 else{
			 singleAcc[currentExp]=-9999;
			 lnbclnccD[currentExp]=-9999;
		 }

		 if (testInstances-determ[currentExp]>0){
			 outputSz[currentExp]/=testInstances-determ[currentExp];
			 setAcc[currentExp]/=testInstances-determ[currentExp];
			 lnbclnccI[currentExp]/=testInstances-determ[currentExp];
		 }

		 else{
			 outputSz[currentExp]=-9999;
			 setAcc[currentExp]=-9999;
			 lnbclnccI[currentExp]=-9999;
		 }
		 determ[currentExp]/=testInstances;
		 discountedAcc[currentExp]/=testInstances;
	 }





	 /** Classifies a single instance, returning the list of predicted classes  
	  */
	 private int[] computeClassification (int[] instance) {
		 int i,j;
		 double test;
		 ArrayList<Integer> undomClasses = new ArrayList<Integer>(numClasses);

		 for (i=0; i<numClasses; i++) {
			 undomClasses.add(i);
		 }

		 for (i=0; i<numClasses; i++)
		 {
			 //if class i is already dominated, no need to check it
			 if (! undomClasses.contains(i)){
				 continue;
			 }


			 for (j=0; j<numClasses; j++)
			 {
				 if (i==j) {
					 continue;
				 }
				 if (! undomClasses.contains(j)){
					 continue;
				 }

				 test=checkCredalDominanceCIR(i,j,instance);

				 if (test > 1){
					 undomClasses.remove(undomClasses.indexOf(j));
				 }
			 }
		 }
		 return ArrayUtils.arrList2Array(undomClasses); 
	 }



	 /**Fills deltaArr,gammaArr, deltaTildeArr, gammaTildeArr, alpha, beta, presentFeats; all these variables
	  * are going to be later used for computing dominance.
	  * 
	  * @return
	  */
	 private void fillArrays(int c1, int c2, int[] instance){
		 deltaTildeArr=new ArrayList<Integer>(numFeats);
		 gammaTildeArr=new ArrayList<Integer>(numFeats);
		 deltaArr=new ArrayList<Integer>(numFeats);
		 gammaArr=new ArrayList<Integer>(numFeats);

		 int i;
		 missingFeats=0;

		 //remember that frequencies of the super class have been computed without any prior additional count, so these casts
		 //do not involve any information loss
		 alpha=(int)classFreqs.get(bwdthIdx)[c1];
		 beta=(int)classFreqs.get(bwdthIdx)[c2];


		 int sum1,sum2,ii;
		 //get the conditional counts 
		 for (i=0; i<numFeats; i++)
		 {	
			 //missing variables are marginalized
			 if (instance[i]==-9999){
				 missingFeats++;
				 continue;
			 }			 

			 gammaArr.add((int)contTables.get(bwdthIdx)[i][instance[i]][c1]);
			 deltaArr.add((int)contTables.get(bwdthIdx)[i][instance[i]][c2]);
			 sum1=0;
			 sum2=0;
			 for (ii=0;ii<numValues[i];ii++){
				 sum1 += contTables.get(bwdthIdx)[i][ii][c1];
				 sum2 += contTables.get(bwdthIdx)[i][ii][c2];
			 }
			 gammaTildeArr.add(sum1);				
			 deltaTildeArr.add(sum2);
		 }			 
	 }

	 /**
	  * Computes the CIR test of dominance between class c1 and c2 (if the returned value is >1, c1 dominates c2)
	  */
	 private double checkCredalDominanceCIR(int c1, int c2, int[] instance){
		 fillArrays(c1,c2, instance);

		 double infHx=-999;
		 //init necessary to compile the code
		 double derivLnH0=0;
		 double derivLnHs;
		 double minimizingX;
		 int i;


		 //if there is j such that n(aM j , cM ) = 0 , inf h(t(cM )) = 0;
		 if (gammaArr.indexOf(0)>=0){			
			 infHx=0;}


		 //NCC2: was if k = 0 and r = 0, inf h(t(cM )) = h(xmax), where k counts the nonMAR feats;
		 else if (missingFeats==gammaArr.size()) {
			 infHx=computeHxCIR(s);}

		 else{

			 //from here on, the function can be assumed to be convex

			 if (beta==0){ 
				 derivLnH0=Double.NEGATIVE_INFINITY;}

			 for (i=0; i<deltaArr.size(); i++)
			 {
				 if ((deltaArr.get(i)==0) & (deltaTildeArr.get(i)>0))
				 {
					 derivLnH0=Double.NEGATIVE_INFINITY;
					 break;
				 }
			 }

			 if (derivLnH0!=Double.NEGATIVE_INFINITY){
				 derivLnH0=computeDerivLnHxCIR (0);}


			 derivLnHs=computeDerivLnHxCIR(s);

			 if (derivLnH0>=0){
				 infHx=computeHxCIR(0);}

			 else if (derivLnHs<=0){
				 infHx=computeHxCIR(s);}

			 else if ((derivLnH0<=0) & (derivLnHs>=0) )
			 {
				 //numerical approximation
				 minimizingX=findZeroCIR(0,s);
				 infHx=computeHxCIR(minimizingX);
			 }
		 }
		 return infHx;
	 }

	 /**Numerical approximation of the min of Ln(Hx) via Newton-Raphson method.
	  */
	 private double findZeroCIR (double x1, double x2)
	 {
		 double eps=Math.pow(10,-7);
		 int j,maxiter=200;
		 double df,dx,dxold,f,fl;
		 double temp,xh,xl,rts;

		 fl=computeDerivLnHxCIR(x1);

		 //Orient the search so that f(xl) < 0.
		 if (fl < 0.0)
		 {	
			 xl=x1;
			 xh=x2;
		 } 
		 else 
		 {
			 xh=x1;
			 xl=x2;
		 }


		 rts=0.5*(x1+x2); //Initialize the guess for root,
		 dxold=Math.abs(x2-x1); //the stepsize ,
		 dx=dxold; //and the last step.


		 f=computeDerivLnHxCIR(rts);
		 df=computeDeriv2LnHxCIR (rts);

		 for (j=1;j<=maxiter;j++) 

		 {
			 if ((((rts-xh)*df-f)*((rts-xl)*df-f) > 0.0) // Bisect if Newton out of range,
					 || (Math.abs(2.0*f) > Math.abs(dxold*df))) // or not decreasing fast enough.
			 {
				 dxold=dx;
				 dx=0.5*(xh-xl);
				 rts=xl+dx;
				 if (xl == rts) {
					 return rts; //Change in root is negligible.
				 }
			 } 
			 //Newton step acceptable. Take it.
			 else { 
				 dxold=dx;
				 dx=f/df;
				 temp=rts;
				 rts -= dx;
				 if (temp == rts) {
					 return rts;
				 }
			 }
			 if (Math.abs(dx) < eps) {
				 return rts; //Convergence criterion.
			 }

			 f=computeDerivLnHxCIR(rts);
			 df=computeDeriv2LnHxCIR (rts);

			 if (f < 0.0) //Maintain the bracket on the root.
				 xl=rts;
			 else {
				 xh=rts;
			 }
		 }
		 System.out.println("Maximum number of iterations exceeded in rtsafe");
		 return rts; //Never get here.
	 }


	 /**Computes  Hx for a given value of x, alpha, beta ecc. (see Corani and Zaffalon, 2007)
	  */
	 private double computeHxCIR(double x)

	 {
		 double tmp;
		 int i;

		 tmp=Math.log(Math.pow((beta+x)/(alpha+s-x),(-1)));

		 for (i=0; i<deltaTildeArr.size(); i++)
		 {
			 tmp += Math.log(deltaTildeArr.get(i) + x) - Math.log(gammaTildeArr.get(i)+s-x);  
			 tmp += Math.log(gammaArr.get(i)) - Math.log(deltaArr.get(i)+x);
		 }
		 return Math.exp(tmp);
	 }


	 /**Computes the derivative of Ln(Hx) (see Corani and Zaffalon, 2007)
	  */

	 private double computeDerivLnHxCIR (double x)

	 {
		 double tmp=0;
		 boolean is_set=false;

		 //if X==0, let's first do some preliminary checks. It might be necessary
		 //setting manually the derivative to -Inf to avoid numerical issues with zeros

		 if (x==0)
		 {

			 if ( (beta==0)) {
				 tmp=Double.NEGATIVE_INFINITY;
				 is_set=true;
			 }
			 else{
				 for (int i=0; i<deltaArr.size(); i++)
				 {
					 if ((deltaArr.get(i)==0) & (deltaTildeArr.get(i)>0)){
						 tmp= Double.NEGATIVE_INFINITY;
						 is_set=true;
						 break;
					 }
				 }
			 }
		 }

		 if (! is_set){
			 //ncc2: (k-1) where k is the number of nonMar features
			 tmp= (-1)/(beta+x);
			 tmp += (-1)/(alpha+s-x);

			 int i;


			 for (i=0; i<deltaArr.size(); i++)
				 tmp += 1/(deltaTildeArr.get(i)+x);

			 for (i=0; i<deltaArr.size(); i++)
				 tmp -= 1/(deltaArr.get(i)+x);

			 for (i=0; i<deltaArr.size(); i++)
				 tmp += 1/(gammaTildeArr.get(i)+s-x);
		 }
		 return tmp;
	 }


	 /**Computes the second derivative of Ln(Hx) (see Corani and Zaffalon, 2007) 
	  */
	 private double computeDeriv2LnHxCIR (double x)
	 {
		 double tmp;		
		 int i;
		 int presentFeats=numFeats-missingFeats;


		 tmp = - (presentFeats-1)/(Math.pow(beta+x,2));		
		 tmp +=(presentFeats-1)/Math.pow((alpha+s-x),2);

		 //		 for (i=0; i<presentFeats; i++)
		 //			 value += 1/Math.pow(betaArr.get(i)+x,2);

		 for (i=0; i<gammaArr.size(); i++)
			 tmp -= 1/Math.pow(deltaTildeArr.get(i)+x,2);

		 for (i=0; i<gammaArr.size(); i++)
			 tmp += 1/Math.pow(deltaArr.get(i)+x,2);

		 for (i=0; i<gammaArr.size(); i++)
			 tmp += 1/Math.pow(gammaTildeArr.get(i)+s-x,2);

		 return tmp;
	 }



	 /**
	  * Returns the matrix of the predictions
	  */
	 public int[][] getPredictions()
	 {
		 return predictions;
	 }

	 /**Writes the supplied value of nbcNccD, computed elsewhere, at
	  * position currentExp in array nbcNccD.
	  */
	 void setNbcNccD(double value){
		 lnbclnccD[currentExp]=value;
	 }

	 /**Writes the supplied value of nbcNccI, computed elsewhere, at
	  * position instanceIdx in array nbcNccI.
	  */
	 void setNbcNccI( double value){
		 lnbclnccI[currentExp]=value;
	 }

	 /**
	  * Returns the vector, which contains the prediction for instance in position idx
	  */
	 int[] getPredictions(int idx)
	 {
		 return predictions[idx];
	 }

	 public  double[] getDeterm() {
		 return determ;
	 }

	 public  double[] getDiscountedAcc() {
		 return discountedAcc;
	 }

	 public  double[] getNbcNccD() {
		 return lnbclnccD;
	 }

	 public  double[] getNbcNccI() {
		 return lnbclnccI;
	 }

	 public  double[] getOutputSz() {
		 return outputSz;
	 }

	 public  double[] getSetAcc() {
		 return setAcc;
	 }

	 public  double[] getSingleAcc() {
		 return singleAcc;
	 }


	 private boolean adaptive;


	 private int maxTrials;

	 /**overall occurrences of class c1*/ 
	 private int alpha;

	 /**overall occurrences of class c2*/ 
	 private int beta;


	 /**gamma array is defined for Mar features only; it contains conditional count with respect to class c1 after having dropped missing data*/
	 private ArrayList<Integer> gammaArr;

	 /**delta array is defined for Mar features only; it contains conditional count with respect to class c2 after having dropped missing data*/
	 private ArrayList<Integer> deltaArr;

	 /**Sum of occurrences of class c1, considering only those instances of the learning set where
		the NonMar feature is non missing. A different value for every feature.*/
	 private ArrayList<Integer> gammaTildeArr;

	 /**Sum of occurrences of class c2, considering only those instances of the learning set where
		the NonMar feature is non missing. A different value for every feature.*/
	 private ArrayList<Integer> deltaTildeArr;


	 /**Number of features missing in current instance*/
	 private int missingFeats;







	 /**Stores NCC predictions; as every prediction can be imprecise and hence contain several
	  * value, it is implemented as a matrix. 
	  * */
	 private int[][] predictions;

	 //static members used to track the performance across different CV runs
	 private double[] singleAcc;
	 private double[] setAcc;
	 private double[] outputSz;
	 private double[] determ;
	 private double[] lnbclnccD;
	 private double[] lnbclnccI;
	 private double[] discountedAcc;
}
