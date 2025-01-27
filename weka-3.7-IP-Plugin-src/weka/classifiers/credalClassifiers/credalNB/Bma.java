package weka.classifiers.credalClassifiers.credalNB;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;

/**
 *Bma implementation, according to Dash & Cooper 2003.
 *Paper retrievable from http://citeseer.ist.psu.edu/580703.html
 */
public class Bma extends NaiveClassifier{

  /** for serialization */
  static final long serialVersionUID = -1478242251770381214L;

/**
 * Computes the parameter of the BMA summary classifier using s=1; if feasible, computes the probability
 * of the different architectures. Compared to the constructor of naiveBayes, the parameter map and 
 * priorCode are missing (map does not make sense with BMA, while priorCode cannot be other than to 2 for BMA).
 */
	void train (ArrayList<int[]> trainingSetPar, ArrayList<String> featNamesPar, ArrayList<String>classNamesPar, 
			ArrayList<Integer> numClassForEachFeature, int sPar, int currentExpPar)
	{
		
		//because BMA does use the global prior
		int priorCode=2;
		
		currentExp=currentExpPar;
		super.train(trainingSetPar, featNamesPar, classNamesPar, numClassForEachFeature,priorCode,sPar);
		computeRho0();
		computeRhoC();
		//writePars();

		//the following code computes the probability of each structure. This is not really needed to compute Bma via D&C algorithm, 
		//yet it is an information of interest. Of course, it can be done only when the numebr fo features is reasonably low.
		computeStructProbabilities();
	}
	
	/**Allocates vector of accuracy and confusion matrix, using the constructor of the super class.
	 */
	Bma(int numExperiments, int numClassesPar)
	{
		super(numClassesPar);
		accuracy= new double[numExperiments];
		significantStructs= new double [numExperiments];
		featuresMap= new double[numExperiments];
	}
	
	
	/**
	 * Writes to file theta.csv the parameters of NBC (computed from rhoC coefficients) and to file thetaStar.csv the parameters
	 * of the summary BMA classifier.
	 */
	void writePars(){
		double[][][] bmaPars;
		int i,j,k;

		
		bmaPars=thetaSummaryClassifier(rhoSummaryClassifier());
		try{
			BufferedWriter out=new BufferedWriter(new FileWriter("/home/giorgio/tmp/thetaBMA .csv",false));
			//header
			out.write(",");
			for (i=0;i<numClasses;i++){
				out.write("c"+i+"(BMA)"+",");
			}
			out.newLine();
			for (i=0;i<numFeats;i++){
				out.write(featureSet[i].getName());
				out.newLine();
				for (k=0;k<numValues[i];k++){
					out.write(",");
					for (j=0;j<numClasses;j++){
						out.write(bmaPars[i][j][k]+",");
					}
					out.newLine();
				}
			}
			out.close();
		}

		catch (IOException e)
		{
			System.out.println("Problems saving parameters to file");
		}
	}

	/**
	 *Computes log(rhoSummary_ijk) as log (rhoC_ijk + rho0_ik), using smartly the logs
	 *to avoid numerical problems 
	 *(note that the computation is not carried out for i=0, as this position refers to the class and not to the features)
	 *The actual values start from position 1 of rhoSummary, to be consistent with rho0 and rhoC.
	 */
	private double[][][] rhoSummaryClassifier(){
		int i,j,k,iPlus;
		double[][][]rhoSummary = new double[numFeats+1][][];

		for (i=0;i<numFeats;i++){
			iPlus=i+1;
			rhoSummary[iPlus]=new double[numClasses][];
			for (j=0;j<numClasses;j++){
				rhoSummary[iPlus][j]=new double[numValues[i]];
			}
		}
		//array allocated


		double sum;
		double shift;

		for (i=0;i<numFeats;i++){
			iPlus=i+1;
			for (j=0;j<outputClasses.length;j++){
				for (k=0;k<numValues[i];k++){
					shift=rhoC[iPlus][j][k];
					if (rho0[iPlus][k]>shift){
						shift=rho0[iPlus][k];
					}
					sum=Math.exp(rhoC[iPlus][j][k]-shift)+Math.exp(rho0[iPlus][k]-shift);
					rhoSummary[iPlus][j][k]=Math.log(sum)+shift;
				}
			}
		}
		
		return rhoSummary;
	}
	
	/**Given a matrix of rho_ijk, stored as logs, returns theta_ijk,
	 * where theta_ijk=rho_ijk/Sum_k(rho_ijk); the theta are actual probabilities and
	 * no longer logs.
	 */
	private double[][][] thetaSummaryClassifier(double[][][] rho_par){
		int i,j,k,iPlus;
		double logSum;
		double shift;
		double pars[][][]=new double[rho_par.length][numClasses][];

		for (i=0;i<numFeats;i++){
			pars[i]=new double[numClasses][];
			for (j=0;j<numClasses;j++){
				pars[i][j]=new double[numValues[i]];
			}
		}
		//array allocated
		
		for (i=0;i<numFeats;i++){
			iPlus=i+1;
			for (j=0;j<outputClasses.length;j++){

				//shift=maximum rho_ijk
				shift=rho_par[iPlus][j][0];
				for (k=1;k<numValues[i];k++){
					if (rho_par[iPlus][j][k]>shift){
						shift=rho_par[iPlus][j][k];
					}
				}
				logSum=0;

				//get the log of sum_k rho_ijk
				for (k=0;k<numValues[i];k++){
					logSum += Math.exp(rho_par[iPlus][j][k]-shift);
				}
				logSum=Math.log(logSum)+shift;


				//computation of \theta_ijk for each k
				for (k=0;k<numValues[i];k++){
					//theta_ijk=exp(log(rho_ijk)-log(Sum(rho_ijk)
					pars[i][j][k]=Math.exp(rho_par[iPlus][j][k]-logSum);
				}
			}
		}
		return pars;		
	}

	
	void classifyInstances(ArrayList<int[]> TestingSet){

		probabilities=new double [TestingSet.size()][numClasses];
		predictions = new int [TestingSet.size()];

		int instCounter,i,j;
		int[] currentInstance;
		double[] tmpLogProbabilities=new double[numClasses];
		double maxLog=Double.MAX_VALUE;

		for (instCounter=0; instCounter<TestingSet.size();instCounter++)
		{
			currentInstance=TestingSet.get(instCounter);
			tmpLogProbabilities=new double[numClasses]; 
			maxLog=-(Double.MAX_VALUE);
			int currentOptClass=-1;

			//compute the probability for each class
			for (j=0;j<numClasses;j++){
				tmpLogProbabilities[j]=computeLogProbability(j, currentInstance);
				if (tmpLogProbabilities[j]>maxLog)
				{
					maxLog=tmpLogProbabilities[j];
					currentOptClass=j;
				}
			}

			//normalize all probabilities to  sum to 1
			for (i=0;i<tmpLogProbabilities.length;i++)
				tmpLogProbabilities[i]=Math.exp(tmpLogProbabilities[i]-maxLog);
			double ProbSum=ArrayUtils.arraySum(tmpLogProbabilities)[0];

			for (j=0; j<numClasses; j++)	{
				probabilities[instCounter][j]=tmpLogProbabilities[j]/ProbSum;
			}
			predictions[instCounter]=currentOptClass;

			//update confusionMatrix
			confMatrix[currentInstance[numFeats]][currentOptClass]++;

			//update accuracy vector and nbcAccurate if accurate
			if (currentOptClass==currentInstance[numFeats]){
				accuracy[currentExp]++;
			}
		}
		//now, correct the accuracy vector
		accuracy[currentExp] /=testInstances;
	}


	/**
	 *Given an instance , computes the log-probability of class classIdx.
	 *The log-probailitiy is computed exploiting the trick \Sum_i exp(l_i) = exp(N) \Sum_i exp(l_i - N),
	 *to overcome numerical issues.
	 *Log-probabilities have to be processed later so that their exponential sum to 1.
	 */
	private double computeLogProbability(int classIdx, int[]currentInstance){

		//here we store the terms of the \product(rho0_i+rhoC_i), which we will finally compute as sum of logarithms
		double logProbability;
		int i;
		double shift;


		logProbability=rhoC[0][classIdx][0];

		//now the actual features
		for (i=0;i<numFeats;i++){
			double[] tmpArr=new double[2];
			int iPlus=i+1;
			
			//this implementation basically ignores missing data
			if (currentInstance[i]!=-9999)
			{
			//rhoC
			tmpArr[0] = rhoC[iPlus][classIdx][currentInstance[i]];		
			shift=tmpArr[0];
			//rho0
			tmpArr[1]=rho0[iPlus][currentInstance[i]];
			if (tmpArr[1]>shift){
				shift=tmpArr[1];
			}
			double sum=(Math.exp(tmpArr[0]-shift)+Math.exp(tmpArr[1]-shift));
			logProbability +=Math.log(sum)+shift;
			}
		}

		return logProbability;
	}


	/**
	 * Computes log of coefficients rhoC
	 */
	private void computeRhoC() {

		double[][][] logThetaC = computeLogThetaC();
		double[] DataLikC = computeDataLikC();


		//0.5 is the prior probability of including a feature in the model, if we adopt a binomial prior over each feature
		//and we set alpha=beta=1. This is p_s(X_i,P_i) in D&C.
		double featPrior = Math.log(0.5);


		rhoC=new double [numFeats+1][][];
		int i,j,k;

		//feature 0 is the class, included with probability 1.
		//we manage it here separately from other features, as it requires different computations.
		rhoC[0]=new double[numClasses][1];

		for (j=0;j<numClasses;j++){
			//we omit here to introduce the prior over the class, which would be log(1)=0 within the brackets
			rhoC[0][j][0]=DataLikC[0]+logThetaC[0][j][0];
		}	

		//now we compute the rho for the actual features

		for (i=0;i<numFeats;i++){
			rhoC[i+1]=new double[numClasses][];
			for (j=0;j<numClasses;j++){
				rhoC[i+1][j]=new double[numValues[i]];
				for (k=0;k<numValues[i];k++)
				{
					rhoC[i+1][j][k]=logThetaC[i+1][j][k]+featPrior+DataLikC[i+1];
				}
			}
		}
	}

	/**
	 * Computes the probability of each struct, and detects how many of them are significant, i.e. are given probability >0.05;
	 * moreover, computes the number of BMA features, as weighted average between the significant structs
	 *
	 */
//	private void detectSignificantStructs(){
//		
//		//how many structures
//		int twoPowN=(int)Math.pow(2, numFeatures);
//		int stepSize=twoPowN/2;
//		int[] currentStructure= new int [numFeatures];
//		int i, j, counter, currentValue;
//		currentValue=1;
//
//		//class feature is present in all structures, so we don't care about is: 
//		//it is constant for all structurers and thus does not affect the relative probability
//		//of the different structures
//
//		for (i=0;i<numFeatures;i++)
//		{
//			counter=0;
//			while (counter<twoPowN)
//			{
//				currentValue *= -1;
//				for (j=0;j<stepSize;j++)
//				{
//					currentStructure[i]=currentValue;
//				}
//			}
//			stepSize/=2;
//		}
//
//	}
	
	/**Computes the probability of each structure; this is notr equired by D&C algorithm, yet it itas interesting to
	 * observe how probability is distributed betweeen structures.
	 */
	private void computeStructProbabilities(){

		double[] dataLik0 = computeDataLik0();
		double[] dataLikC = computeDataLikC();

		featuresMap[currentExp]=0;
		for (int i=1; i<numFeats+1; i++){
			if (dataLikC[i]>dataLik0[i])
				featuresMap[currentExp]++;
		}

		if (numFeats>15) {
			significantStructs[currentExp]=-9999;
		}
		else
		{
			int[][]structures=computeStructures();
			double[] structProbabilities=new double[structures.length];

			int i,j;
			for (i=0; i<structProbabilities.length;i++){
				structProbabilities[i]=0;
				for (j=0;j<(structures[i].length);j++)
				{
					if (structures[i][j]==1)
						structProbabilities[i]+=dataLikC[j];
					else if (structures[i][j]==-1)
						structProbabilities[i]+=dataLik0[j];
				}
			}

			double maxLog=-(Double.MAX_VALUE);

			//compute the probability for each class
			for (j=0;j<structProbabilities.length;j++){
				if (structProbabilities[j]>maxLog)
					maxLog=structProbabilities[j];
			}
			for (j=0;j<structProbabilities.length;j++)
				structProbabilities[j]=Math.exp(structProbabilities[j]-maxLog);

			//normalize to sum1
			double sum=ArrayUtils.arraySum( structProbabilities)[0];
			significantStructs[currentExp]=0;
			for (j=0;j<structProbabilities.length;j++){
				structProbabilities[j]/=sum;
				if ((structProbabilities[j])>0.05){
					significantStructs[currentExp]++;
				}
			}
		}
	}
	
	
	/**
	 * NOT NEEDED BY D&C algorithm.
	 * Computes all the possible structures (each represented by a string of -1 and 1), given a set of attributes;
	 *there are 2^n possible structures.
	 */
	private int[][] computeStructures()
	{
		int twoPowN=(int)Math.pow(2, numFeats);

		int[][] structures=new int [twoPowN][numFeats+1];
		int stepSize=twoPowN/2;

		int i, j, counter, currentValue;
		currentValue=1;

		//class feature
		for (i=0;i<twoPowN;i++)
			structures[i][0]=1;

		for (i=0;i<numFeats;i++)
		{
			counter=0;
			while (counter<twoPowN)
			{
				currentValue *= -1;
				for (j=0;j<stepSize;j++)
				{
					structures[counter++][i+1]=currentValue;
				}
			}
			stepSize/=2;
		}
		return structures;
	}



	/**
	 * Computes log of coefficients rho0
	 */
	private void computeRho0(){
		double[][] logTheta0 = computeLogTheta0();
		double[] dataLik0 = computeDataLik0();

		//0.5 is the prior probability of including a feature in the model, if we adopt a binomial prior over each feature
		//and we set alpha=beta=1. This is p_s(X_i,P_i) in D&C.
		double featPrior = Math.log(0.5);

		rho0=new double [numFeats+1][];
		int i,k;

		//feature 0 is the class, included with probability 1.
		//we manage it here separately from other features, as it requires different computations.

		//note that rho0[0] will never be accessed, as the class is included with probability 1. 
		//rho0[0]=new double[numClasses][1];

		//now we compute the rho for the actual features
		for (i=0;i<numFeats;i++){
			int iPlus=i+1;
			rho0[iPlus]=new double[numValues[i]];
			for (k=0;k<numValues[i];k++)
			{
				rho0[iPlus][k]=logTheta0[iPlus][k]+featPrior+dataLik0[iPlus];
			}
		}
	}


	void saveRatios(String fileAddress)
	{
		String resultsFile = fileAddress+"Bma.csv";
		DecimalFormat formatter= new DecimalFormat("#0.000");
		try{
			int i,j;
			BufferedWriter out=new BufferedWriter(new FileWriter(resultsFile,false));
			for (i=0; i<probabilities.length;i++)
			{
				for (j=1; j<probabilities[0].length-1;j++)
					out.write(formatter.format(probabilities[i][0]/probabilities[i][j])+"\t");
				out.write(formatter.format(probabilities[i][0]/probabilities[i][j])+"\n");
			}
			out.close();
		}

		catch (IOException e)
		{
			System.out.println("Problems saving probabilities to file");
		}		
	}

	
	
//	GETTERS
	public int[] getPredictions() {
		return predictions;
	}
	
	public double[][] getEstimatedProbabilities() {
		return probabilities;
	}
	
	public double[] getSignificantStructs() {
		return significantStructs;
	}
	
	public double[] getFeaturesMap(){
		return featuresMap;
	}
	
	public double[] getAccuracy() {
		return accuracy;
	}


	//==Data members
	double[][][] rhoC;
	double[][] rho0;

	

	/**Num of structures with more than 5% probability */
	//double just for simplicity in writeResults 
	double[] significantStructs;

	/**Weigthed average of number of features  in significant structures*/
	double[] featuresMap;
	
	/**Accuracy as measured over different experiments (for instance CV)*/
	private  double[] accuracy;


	


	/**predicted classes*/
	int[] predictions;

}