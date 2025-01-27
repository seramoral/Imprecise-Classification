package weka.classifiers.credalClassifiers.credalNB;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Implements the Naive Bayes Classifier (NBC) with Laplace prior
 */

public class NaiveBayes extends NaiveClassifier{

      /** for serialization */
      static final long serialVersionUID = -1478242251770381214L;

	/**
	 * Initializes  features and computes log-probabilities, thus training the classifier;
	 * if map, identifies the maximum a posteriori architecture.
	 */
	void train (ArrayList<int[]> trainingSet, ArrayList<String> featureNames, ArrayList<String>classNames, 
			ArrayList<Integer> numClassForEachFeature,  int priorCode, 
			boolean map, int sPar){

		super.train(trainingSet, featureNames, classNames, numClassForEachFeature,priorCode,sPar);


		//identify the MAP architecture if necessary
		if  (map){
			mapArchitecture= new int [numFeats];
			Arrays.fill(mapArchitecture, 1);
			double[] dataLik0 = computeDataLik0();
			double[] dataLikC = computeDataLikC();
			for (int i=1; i<numFeats+1; i++){
				if (dataLik0[i]>dataLikC[i])
					mapArchitecture[i-1]=-1;
			}
		}
	}
	
	/**Initializes accuracy, numClasses and confMatrix (all these variables are persistent
	 * across the cross-validation runs); all remaining data member will be dealt 
	 * with by function train.
	 * 
	 */
	NaiveBayes(int numExperiments,int numClassesPar){
		super(numClassesPar);
		accuracy= new double[numExperiments];
	}
	
	
	
	/**Classifies all the instances of the supplied TestingSet, writing the results of the computation into
	 * EstimatedProbabilities and PredictedInstances
	 * @param currentExpPar 
	 */
	void classifyInstances(ArrayList<int[]> TestingSet, int currentExpPar)
	{
		currentExp=currentExpPar;
		predictions= new int[TestingSet.size()];
		probabilities=new double[TestingSet.size()][numClasses];
		testInstances=TestingSet.size();
		
		int i;

		for (i=0; i<TestingSet.size(); i++){
			classifyInstance(TestingSet.get(i), i);
		}

		//now, correct the accuracy vector
		accuracy[currentExp] /=testInstances;
	}


	/**
	 * Writes to file theta.csv the parameters of NBC (computed from rhoC coefficients) and to file thetaStar.csv the parameters
	 * of the summary BMA classifier.
	 */
	void writePars(){
		
		try{
			int i,k,j;
			BufferedWriter out=new BufferedWriter(new FileWriter("/home/giorgio/tmp/thetaNBC .csv",false));
			//header
			out.write(",");
			for (i=0;i<numClasses;i++){
				out.write("c"+i+"(NBC)"+",");
			}
			out.newLine();
			for (i=0;i<numFeats;i++){
				out.write(featureSet[i].getName());
				out.newLine();
				for (k=0;k<numValues[i];k++){
					out.write(",");
					for (j=0;j<numClasses;j++){
						out.write(Math.exp(featureSet[i].getLogProbability(j)[k])+",");
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
	 * Classify a single instance, writing the computed probabilities at position InstanceIdx
	 * of probabilities, and the predicted class at position InstanceIdx of predictions.
	 * Update confusion matrix and accuracy vector.
	 */
	private void classifyInstance (int[] suppliedInstance, int  instanceIdx)
	{
		int i,j; 
		//double logProb;
		double[] logProbArray=new double[numClasses];
		double maxLogProb=-Double.MAX_VALUE;
		int maxProbIdx=-1;

		//probabilities of each class
		for (i=0; i<numClasses; i++)
		{

			logProbArray[i]=0;

			//get a priori log-probability
			logProbArray[i]+=outputClasses[i].getLogProbability();

			//add conditioned log-probability
			for (j=0; j<numFeats; j++)
			{
				//marginalization for missing data
				if (suppliedInstance[j] == -9999){
					continue;
				}
				//we use && so that if mapArchitecture is not instantiated, the remaining conditions is not evaluated
				if (mapArchitecture!=null && mapArchitecture[j]==-1){
					continue;
				}

				logProbArray[i]+=featureSet[j].getLogProbability(i,suppliedInstance[j]);

			}//==Log probability computed


			if (logProbArray[i]>maxLogProb)
			{
				maxLogProb=logProbArray[i];
				maxProbIdx=i;
			}
		}
		predictions[instanceIdx] = maxProbIdx;

		//****compute the probabilities in a numerical robust way

		//first, compute shift
		double shift=logProbArray[maxProbIdx];
		double sumProb=0;
		double[] tmpArr = new double [numClasses];

		for (i=0; i<numClasses; i++)	{
			tmpArr[i]=Math.exp(logProbArray[i]-shift);
			sumProb+=tmpArr[i];
		}

		for (i=0; i<numClasses; i++)	
			probabilities[instanceIdx][i]=tmpArr[i]/sumProb;

		//update confusionMatrix
		confMatrix[suppliedInstance[numFeats]][maxProbIdx]++;
		
		//update accuracy vector and nbcAccurate if accurate
		if (maxProbIdx==suppliedInstance[numFeats]){
			accuracy[currentExp]++;
		}

	}//==END classify instance

	int[] getPredictions() {
		return predictions;
	}	

	public double[][] getProbabilities() {
		return probabilities;
	}
	
	public int[][] getNbcConfMatrix(){
		return confMatrix;
	}

	public  double[] getAccuracy() {
		return accuracy;
	}
	
	public int[] getMapArchitecture(){
		return mapArchitecture;
	}
	
	
	//DATA MEMBERS
	/**
	 * Index of the class predicted for each instance
	 */
	private int[] predictions;
	
	/**
	 * Maximum a posteriori architecture: +1 indicates present features, -1 indicates dropped features
	 */
	private int[] mapArchitecture;

	/**Accuracy as measured over different experiments (for instance CV)*/
	private  double[] accuracy;
	
}
