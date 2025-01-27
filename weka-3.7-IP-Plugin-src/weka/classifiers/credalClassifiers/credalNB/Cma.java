package weka.classifiers.credalClassifiers.credalNB;


import java.util.ArrayList;
import java.util.Arrays;

/**
 *Cma implementation
*/
public class Cma extends NaiveClassifier{

        /** for serialization */
        static final long serialVersionUID = -1478242251770381214L;

	public Cma (int numExperiments,int numClassesPar)
	{
		super(numClassesPar);
		singleAcc=new double[numExperiments];
		discountedAcc=new double[numExperiments];
		setAcc=new double[numExperiments];
		outputSz=new double[numExperiments];
		determ=new double[numExperiments];
		bmaCmaD=new double[numExperiments];
		bmaCmaI=new double[numExperiments];
	}

	public void train(ArrayList<int[]> trainingSetPar, ArrayList<String> featNamesPar, ArrayList<String>classNamesPar, ArrayList<Integer> numClassForEachFeature,
			int sPar)
	{

		int priorCode=2;
		super.train(trainingSetPar, featNamesPar, classNamesPar, numClassForEachFeature,priorCode,sPar);


		logTheta0 = computeLogTheta0();
		logDataLik0 = computeDataLik0();
		logThetaC = computeLogThetaC();
		logDataLikC = computeDataLikC();
		tmin=0.1;
}
	
	/**
	 * ==This is almost a copy of the same method in the naiveCredal class; would be
	 * nice to find a more elegant solution.==
	 * At the and of a single training/testing experiment, computes
	 *the actual indicators starting from the raw information which is stored in
	 *different arrays. Fills then the ultimate values of such indicators into the
	 *arrays.
	 */
	private void computePerfIndicators(){

		if (determ[currentExp]>0){
			singleAcc[currentExp]/=determ[currentExp];
			bmaCmaD[currentExp]/=determ[currentExp];
		}
		else{
			singleAcc[currentExp]=-9999;
			bmaCmaD[currentExp]=-9999;
		}

		if (testInstances-determ[currentExp]>0){
			outputSz[currentExp]/=testInstances-determ[currentExp];
			setAcc[currentExp]/=testInstances-determ[currentExp];
			bmaCmaI[currentExp]/=testInstances-determ[currentExp];
		}

		else{
			outputSz[currentExp]=-9999;
			setAcc[currentExp]=-9999;
			bmaCmaI[currentExp]=-9999;
		}
		determ[currentExp]/=testInstances;
		discountedAcc[currentExp]/=testInstances;
	}
	
	


	public void classifyInstances(ArrayList<int[]> testingSet, int currentExpPar){

		currentExp=currentExpPar;
		predictions = new int [testingSet.size()][];
		int instCounter;


		for (instCounter=0; instCounter<testingSet.size();instCounter++)
		{
			currentInstance=testingSet.get(instCounter);
			predictions[instCounter]=classifyInstance(currentInstance);
			updatePerfStats(instCounter, currentInstance[numFeats]);
		}
		computePerfIndicators();
	}

	/*By design, this function should be private and moreover it would not need any parameter, as currentInstance is
	 * already a data member of the class.
	 * Making it public and letting the int[] instance as parameter is however necessary to allow calling from Weka.
	 */
	public int[] classifyInstance(int[] instance){
		int c1,c2,i;
		boolean dominated;
		
		currentInstance=instance;
		currentlyUndominated=new ArrayList<Integer>(numClasses);
		
		for (i=0;i<numClasses;i++){
			currentlyUndominated.add(i);
		}

		//run the test of dominance for every feasible pair c1,c2
		for (c1=0;c1<numClasses;c1++){

			if(! currentlyUndominated.contains(c1)){
				continue;
			}
			for (c2=0; c2<numClasses; c2++){
				if (c1==c2){
					continue;
				}
				if (! currentlyUndominated.contains(c2)){
					continue;
				}
				dominated=checkDominance(c1,c2);
				if (dominated){
					currentlyUndominated.remove(currentlyUndominated.indexOf(c2));
				}
			}
		}
		return ArrayUtils.arrList2Array(currentlyUndominated);
	}

	/**
	 * Given two classes c1 and c2, returns true if c1 dominates c2 and false otherwise 
	 */
	boolean checkDominance (Integer c1, Integer c2){
		int featIdx ;

		//the array collect, for each feature, the value of t which minimizes the objective function
		//the first feature, indexed by 0, is the class, for which t is known to be 1  a priori.
		double[] tArr=new double[numFeats+1];
		tArr[0]=1;

		for (featIdx=1;featIdx<tArr.length;featIdx++)
		{
			//skip missing features. Remember that currentInstance indexes features startign from 0
			if (currentInstance[featIdx-1]==-9999){
				continue;
			}
			tArr[featIdx]=computeTopt(c1,c2,featIdx);
		}
		
		//and now compute the value the function
		double logMin=0;
		
		//contribution of unconditioned probabilities coming from the class
		logMin += logThetaC[0][c1][0];
		logMin -= logThetaC[0][c2][0];
		
		//remember that feature 0 is the class
		for (featIdx=1;featIdx<numFeats+1;featIdx++){
			//skip missing features.
			if (currentInstance[featIdx-1]==-9999){
				continue;
			}
			logMin += computeLogFunctionFloor(c1,featIdx,tArr[featIdx]);
			logMin -= computeLogFunctionFloor(c2,featIdx,tArr[featIdx]);
			}
		
		//debug code
//		String tmp="/home/giorgio/tmp/credal.csv";
//		if (c1==0)
//		try{
//			BufferedWriter out=new BufferedWriter(new FileWriter(tmp,true));
//			if (c2==1)
//				out.write("\n"+Math.exp(logMin));
//			else
//				out.write("\t"+Math.exp(logMin));
//			out.close();
//		}
//		catch (IOException e)
//		{
//			System.out.println("Problems saving probabilities to file");
//		}
//		//end debug code
		
		return (Math.exp(logMin)>1);
	}
	
	/**
	 * After having classified a single instance, stores temporary information about
	 * whether it was accurate, determinate etc.
	 */	
	private void updatePerfStats(int instanceIdx, int realClass){

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
		//eventually, update multiLabel Acc
		if (Arrays.binarySearch(predictions[instanceIdx],realClass)>= 0){
			discountedAcc[currentExp] += (double)1/predictions[instanceIdx].length;
		}
	}
	
	/**
	 * Computes the log of a  "floor" of the objective function, referring to class classIdx.
	 * Note that numerator and denominator of the objective fucntion are defined identically, 
	 * they only differ as for the index of the class; therefore the same function can compute both the 
	 * numerator and the denominator; it only matters that the correct classIdx is provided.
	 */
	
	double computeLogFunctionFloor(int classIdx, int featIdx, double t){
		double shift;
		//remember that in currentinstance features are indexed starting from 0, while in logTheta and in logData from 1
		double a =logThetaC[featIdx][classIdx][currentInstance[featIdx-1]]+logDataLikC[featIdx];
		shift=a;
		double b=logTheta0[featIdx][currentInstance[featIdx-1]]+logDataLik0[featIdx];
		
		if(b>shift){
			shift=b;
		}
		return Math.log(t*Math.exp(a-shift)+(1-t)*Math.exp(b-shift))+shift;
	}
	
	
	

	/**Computes the value of t (prior probability of presence of feature i in the current structure) which minimizes the 
	 */
	double computeTopt(int c1, int c2, int featureIdx){
		
		
		
		double t;
		double thetaC1, thetaC2;

		//remember that in currentInstance features are indexed strating from 0,
		//while in logTheta they are indexed starting from 1
		thetaC1=logThetaC[featureIdx][c1][currentInstance[featureIdx-1]];
		thetaC2=logThetaC[featureIdx][c2][currentInstance[featureIdx-1]];

		if (thetaC1>thetaC2){
			t=tmin;
		}
		else{
			t=1-tmin;
		}
		return t;
	}


	//	GETTERS
	public int[][] getPredictions() {
		return predictions;
	}
	public double[] getBmaCmaD() {
		return bmaCmaD;
	}

	public void setBmaCmaD(double bmaCmaD) {
		this.bmaCmaD[currentExp] = bmaCmaD;
	}

	public double[] getBmaCmaI() {
		return bmaCmaI;
	}

	public void setBmaCmaI(double bmaCmaI) {
		this.bmaCmaI[currentExp] = bmaCmaI;
	}

	public double[] getSingleAcc() {
		return singleAcc;
	}

	public double[] getDiscountedAcc() {
		return discountedAcc;
	}

	public double[] getSetAcc() {
		return setAcc;
	}

	public double[] getOutputSz() {
		return outputSz;
	}

	public double[] getDeterm() {
		return determ;
	}



	//==Data members


	/**predicted classes*/
	int[][] predictions;

	/**Current instance to be classified*/
	int[] currentInstance;

	/**minimum value alloed for t (prior probability of presence of every singel feature) */
	double tmin;

	double[][] logTheta0;
	double[] logDataLik0;
	double[][][] logThetaC;
	double[] logDataLikC;
	double[] singleAcc;
	double[] discountedAcc;
	double[] setAcc;
	double[] outputSz;
	double[] determ;
	double[] bmaCmaD;
	double[] bmaCmaI;


	/**Sets of classes currently undominated (to be rebuilt before the classification of any single instance, and then made smaller by
	 * the classification procedure)*/
	ArrayList<Integer> currentlyUndominated;
}