/**
 * NaiveClassifier.java
 * @author Giorgio Corani (giorgio@idsia.ch)
 * 
 * Copyright:
 * Giorgio Corani, Marco Zaffalon
 *
 * IDSIA
 * Istituto Dalle Molle di Studi sull'Intelligenza Artificiale
 * Manno, Switzerland
 * www.idsia.ch
 *
 * The JNCC distribution is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License as published by the Free Software 
 * Foundation (either version 2 of the License or, at your option, any later
 * version), provided that this notice and the name of the author appear in all 
 * copies. JNCC is distributed "as is", in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for 
 * more details.
 * You should have received a copy of the GNU General Public License
 * along with the JNCC distribution. If not, write to the Free
 * Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

package weka.classifiers.credalClassifiers.credalNB;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;

/**Abstract super-class for Naive Classifiers 
 * 
 */
abstract public class NaiveClassifier implements Serializable{

      /** for serialization */
      static final long serialVersionUID = -1478242251770381214L;
	
	/**
	 * Initializes all features and output classes; computes all the relevant conditionalFreq on the training set, setting the specified prior
	 * (0:0; 1:laplace; 2:uniform) 
	 */
	NaiveClassifier(int numClassesPar)
	{	
		//init of various data members
		numClasses=numClassesPar;
		confMatrix=new int [numClasses][numClasses];
	}
	
	/**
	 * Initializes  features and computes log-probabilities, thus training the classifier.
	 */
	void train (ArrayList<int[]> trainingSet, ArrayList<String> featureNames, ArrayList<String>classNames, 
			ArrayList<Integer> numClassForEachFeature,  int priorCode, int sPar){

		//init of various data members
		numFeats=featureNames.size();
		trainInstances=trainingSet.size();
		numValues=numClassForEachFeature.toArray(new Integer[] {});
		s=sPar;

		
		//all prior counts initialized
		setPriorCoefficients(priorCode);
		buildOutputClasses(trainingSet, classNames);
		buildFeatureSet (trainingSet, featureNames, numClassForEachFeature);

	}
	
	
	/**Defines the coefficients that will be used to initializes the count:
	 * 1 everywhere for Laplace, different coeffs for global prior (priorCode 2) or
	 * m-Laplace (priorCode 3).
	 * 
	 */
	void setPriorCoefficients(int priorCode){
		//prior counts for classes, conditional and unconditional frequencies
		pcCond=new double [numFeats];
		pcUncond=new double [numFeats];

		if (priorCode==0 | priorCode==1){
			pcClass=priorCode*s;
			for (int i=0; i<numFeats; i++){
				pcCond[i]=priorCode*s; 
				pcUncond[i]=priorCode*s; 
			}
		}

		else	if (priorCode==2){
			pcClass=(double)s/numClasses; 
			for (int i=0; i<numFeats; i++){
				pcCond[i]=(double)s/(numClasses*numValues[i]); 
				pcUncond[i]=(double)s/(numValues[i]); 
			}
		}
		
		else	if (priorCode==3){
			pcClass=(double)1/(trainInstances); 
			for (int i=0; i<numFeats; i++){
				pcCond[i]=(double)1/(trainInstances); 
				pcUncond[i]=(double)1/(trainInstances); 
			}
		}

		else {
			System.out.println("Wrong prior type specified");
			System.exit(0);
		}
		
		
	}

	
	 /**Writes confusion matrix to file, preceeding it by a title which depends on the classifier
	  * parameter (nbc or ncc2); the confusion matrix is appended into a an already existing file (if the file is
	  * not existing, it is created)
	 * @param numCvRuns 
	  */
	protected void writeConfMatrix(String confFile, String datasetName, int[][] matrix, int numCvRuns){


		int i,j;

		try{
			BufferedWriter out=new BufferedWriter(new FileWriter(confFile,true));

			//write 4 white lines if nbc, to separate from previous experiment
			if (this.getClass().getName().equalsIgnoreCase("NaiveBayes")){	
				for (i=1;i<4; i++){
					out.newLine();
				}
			}
			out.newLine();
			out.write("Dataset: "+datasetName);
			out.newLine();

			//write classifier
			if (this.getClass().getName().equalsIgnoreCase("jncc20.NaiveBayes")){
				out.write("Classifier: NBC");
				out.newLine();
			}
			else if (this.getClass().getName().equalsIgnoreCase("jncc20.NaiveCredalClassifier2")){
				out.write("Classifier: NCC2 (matrix refers to determinate instances)");
				out.newLine();
			}

			for (i=0; i<outputClasses.length; i++){
				out.write(outputClasses[i].getName()+"\t");
			}
			out.write("\t <--classified as");
			out.newLine();

			//note that currentExp contains at this point the overall total number of performed experiments
			
			if (currentExp>0)
			{	
				DecimalFormat formatter= new DecimalFormat("#0");
				for (i=0; i<matrix.length; i++)
				{
					for (j=0; j<matrix[i].length-1; j++)
						out.write(formatter.format(Math.round((double)(matrix[i][j]/(numCvRuns))))+"\t"); 
					out.write(formatter.format (Math.round((double)(matrix[i][j]/(numCvRuns))))
							+"\t"+outputClasses[i].getName());
					out.newLine();
				}
			}

			//if currentExp==0, it means we have performed a single training-testing experiment.
			//We can hence simply write the confMatrix as it is.
			else
			{
				for (i=0; i<matrix.length; i++)
				{
					for (j=0; j<matrix[i].length-1; j++){
						out.write(matrix[i][j]+"\t");}
					out.write(matrix[i][j]+"\t"+outputClasses[i].getName());
					out.newLine();
				}
			}
			out.close();
			}

			catch (IOException ioexc) {
				System.out.println("Unexpected exception writing Confusion Matrix file " +  confFile);
				System.out.println("Please check directory permission.");
				System.exit(0);
			} 
		}
	
	/**
	 * Computes the logarithm of theta_hat for feature i, class j and value k of the feature;
	 * theta_hat is (n_ijk+1/(nc*nai))/(n_cj+1/nc).
	 * Feature 0 is the class and in that case (i=0) the computation simplies to (n_j+1/(nc))/(n+1). 
	 * 
	 */
	protected double[][][] computeLogThetaC(){

		double[][][] logThetaC = new double[numFeats+1][numClasses][];

		//counters of the feature, the class and the feature value
		int i,j,k,kk;

		//i=0 ==> the feature is the class
		//classes are always present, so we simply using trainInstances for the division
		for (j=0;j<numClasses;j++){
			logThetaC[0][j]=new double[1];
			logThetaC[0][j][0]=Math.log(((double)outputClasses[j].frequency)/((double)1+trainInstances));
		}

		//here the actual features
		for (i=0;i<numFeats;i++)
		{
			for (j=0;j<numClasses;j++){
				logThetaC[i+1][j]=new double[numValues[i]];
				//be aware that we cannot simply divide by the number of train instances+1.
				//in fact, we have to manage dataset containing missing data, so the correct way of computing the probability
				//is to divide using the total number of instances in which the feature is present and the class is the one with respect to which we want to condition
				double instancesK=0;
				for (kk=0;kk<numValues[i];kk++){
					instancesK+=featureSet[i].getConditionalFrequencies(j,kk);
				}
				
				for (k=0;k<numValues[i];k++){
					logThetaC[i+1][j][k]=Math.log(((double)featureSet[i].getConditionalFrequencies(j,k))/instancesK);
				}
			}
		}

		return logThetaC;
	}




	/**
	 * Computes the logarithm of theta_hat for feature i, and value k of the feature; it is assumed that the feature is
	 * not included in the architecture, so it is not conditioned on the class.
	 * The counts are hence theta_hat = (n_ai+(1/nai))/(n+1).
	 * Note that the class is included with probability 1, and therefore the Theta0 coefficients are empty for feature 0 (the class).
	 * Yet, we insert these empty coefficients int he matrix, to allow an access to the data coherent to that of ThetaC.
	 */
	protected double[][] computeLogTheta0(){

		int i, k,kk;
		double[][] logTheta0 = new double[numFeats+1][];

		for (i=0;i<numFeats;i++)
		{
			logTheta0[i+1]=new double[numValues[i]];
			for (k=0;k<numValues[i];k++){
				
				//be aware that we cannot simply divide by the number of train instances+1.
				//in fact, we have to manage dataset containing missing data, so the correct way of computing the probability
				//is to divide using the total number of instances in which the feature is present
				double instancesK=0;
				for (kk=0;kk<numValues[i];kk++){
					instancesK+=featureSet[i].getUncondFrequencies()[kk];
				}
				logTheta0[i+1][k]=Math.log((featureSet[i].getUncondFrequencies()[k])/instancesK);
			}
		}
		return logTheta0;
	}






		/**
		 * Instantiates the FeatureSet, by 
		 * computing all the relevant conditionalFreq of all features on the training set; note that a Laplace prior can be introduced, by setting
		 * parameter priorCounts different from zero (i.e., the quantity priorCounts will be then added to each computed count).  
		 * In particular, it computes:<p>
		 * the bivariates count n(a_i,c_j), that correspond to the occurences ignoring missing data for NBC,
		 * and to the lower counts for NCC;<p>
		 * for each output class, the number of missing data of the current feature, needed to then compute the upper counts
		 * for NCC. 
		 * The priorType defines the prior to be used (0:0; 1:laplace; 2:uniform) .
		 * 
		 * 
		 * <p>
		 */
		protected void buildFeatureSet (ArrayList<int[]> TrainingSet, ArrayList<String> FeatureNames, 
				ArrayList<Integer> NumClassForEachFeature) {

			//last column in TrainingSet contains output classes
			int i,j;
			int currentFeatValue, currentClass;
			double[][] Frequencies;
			int[] Missing;
			featureSet=new Feature[FeatureNames.size()];

			//loop over all features
			for (i=0; i<numFeats; i++)
			{		
				Frequencies = new double[numClasses][NumClassForEachFeature.get(i)];
				Missing = new int[numClasses];

				for (double[] ArrayRow :Frequencies){
					Arrays.fill(ArrayRow,pcCond[i]);
				}

				Arrays.fill(Missing,0);

				//scan all the rows of the training set
				for (j=0; j<trainInstances; j++)
				{
					currentFeatValue=TrainingSet.get(j)[i];
					currentClass=TrainingSet.get(j)[numFeats];

					//non-missing datum
					if (currentFeatValue != -9999){
						Frequencies[currentClass][currentFeatValue]++;
					}
					else{
						Missing[currentClass]++;
					}
				}

				featureSet[i]= new Feature (FeatureNames.get(i), Frequencies, Missing); 
			}



		}//==END buildFeatureSet



		/**
		 *Instantiates class names and conditionalFreq of the OutputClass;  prior is defined by parameter priorType (0:0; 1:laplace; 2:uniform)
		 */
		protected  void buildOutputClasses(ArrayList<int[]> TrainingSet, ArrayList<String >ClassNames)
		{
			int i;
			int currentClass;
			double[] ClassFrequencies= new double [numClasses];

			Arrays.fill(ClassFrequencies,pcClass);

			//class is in the last position within the traning set
			int LastIdx=(TrainingSet.get(1).length)-1;

			for (i=0; i<trainInstances; i++)
			{
				currentClass=TrainingSet.get(i)[LastIdx];
				ClassFrequencies[currentClass]++;
			}


			//Now, compute the a priori probability of each class
			double LogProbability[]=new double[numClasses];
			double CountSum=ArrayUtils.arraySum(ClassFrequencies)[0];


			for (i=0; i<ClassFrequencies.length; i++)
			{
				LogProbability[i]=Math.log((double)ClassFrequencies[i]/CountSum);
			}

			outputClasses=new OutputClass[numClasses];
			for (i=0; i<numClasses;i++)
			{
				outputClasses[i]= new OutputClass (ClassNames.get(i), ClassFrequencies[i], LogProbability[i]);
			}

		}

		/**The gamma function is necessary in order to compute the marginal likelihood. 
		 * Code taken from  StatsconLib.java
	see: www.symbolicnet.org/conferences/iamc02/IAMCNosal.pdf
    Function gammaln: returns the value of ln(gamma(xx)) for xx > 0*
		 */
		double gammaln(double xx) {
			double x, tmp, ser;
			double cof[] = {76.18009173, -86.50532033, 24.01409822, -
					1.231739516, 0.120858003e-2, -0.536382e-5};
			int j;
			x = xx - 1.0;
			tmp = x + 5.5;
			tmp-= (x+0.5)*Math.log(tmp);
			ser = 1.0;
			for(j = 0; j <= 5; j++) {
				x += 1.0;
				ser+= cof[j]/x;
			}
			return -tmp+Math.log(2.50662827465*ser);
		}
		
		/**
		 * Computes (in log form) the term related to the likelihood of the data in the expression of rhoC.
		 * See equation (8) of D&C paper.
		 * 
		 */
		protected double[] computeDataLikC(){

			double[] dataLikC = new double[numFeats+1];

//			if (weightSchema.equalsIgnoreCase("uniform")) {
//			Arrays.fill(dataLikC, 1);
//			}
//			else{
			Arrays.fill(dataLikC, 0);

			//counters of the feature, the class and the feature value
			int i,j,k,kk;

			//i=0 ==> the feature is the class
			dataLikC[0]+= gammaln(1)-gammaln(1+trainInstances);
			for (j=0;j<numClasses;j++){
				dataLikC[0]+= gammaln(outputClasses[j].getFrequency())-gammaln(pcClass);				
			}

			//here the actual features
			for (i=0;i<numFeats;i++)
			{

				for (j=0;j<numClasses;j++){

					double instancesK=0;
					for (kk=0;kk<numValues[i];kk++){
						instancesK+=(double)featureSet[i].getConditionalFrequencies(j,kk);
					}
					dataLikC[i+1] += gammaln(pcClass)-gammaln(instancesK);

					for (k=0;k<numValues[i];k++){
						dataLikC[i+1] += gammaln(featureSet[i].getConditionalFrequencies(j,k))-gammaln(pcCond[i]);
					}
				}
			}
//			}
			return dataLikC;
		}

		/**
		 * Computes (in log form) the term related to the likelihood of the data in the expression of rho0.
		 * See equation (8) of D&C paper.
		 * Of course computeDataLik0[0] will be never accessed, as the class is present in the classifier with probability 1.
		 * 
		 */
		protected double[] computeDataLik0(){

			double[] dataLik0 = new double[numFeats+1];
			Arrays.fill(dataLik0, 0);

			//counters of the feature, the class and the feature value
			int i,k,kk;


			//here the actual features
			for (i=0;i<numFeats;i++)
			{
				//be aware that we cannot simply use the number of train instances+1 in gamma(1)/gamma(nl).
				//in fact, we have to manage dataset containing missing data, so the correct way of computing this ratio is to use as argument of the gamma at
				//denominator the total number of instances in which the feature is present
				double instancesK=0;
				for (kk=0;kk<numValues[i];kk++){
					instancesK+=featureSet[i].getUncondFrequencies()[kk];
				}
				dataLik0[i+1] += gammaln(1)-gammaln(instancesK);
				for (k=0;k<numValues[i];k++){
					dataLik0[i+1] += gammaln(featureSet[i].getUncondFrequencies()[k])-gammaln(pcUncond[i]);
				}
			}
			return dataLik0;
		}

		void saveProbabilities(String fileAddress)
		{
			DecimalFormat formatter= new DecimalFormat("#0.000");
			try{
				int i,j;
				BufferedWriter out=new BufferedWriter(new FileWriter(fileAddress,false));
				for (i=0; i<probabilities.length;i++)
				{
					for (j=0; j<probabilities[0].length-1;j++)
						out.write(formatter.format(probabilities[i][j])+"\t");
					out.write(formatter.format(probabilities[i][j])+"\n");
				}
				out.close();
			}

			catch (IOException e)
			{
				System.out.println("Problems saving probabilities to file");
			}		
		}

		//==GETTERS
		public OutputClass[] getOutputClasses() {
			return outputClasses;
		}



		//==DATA MEMBERS
		/**
		 * Array of Feature objects, that represents the feature set of the classifier
		 */
		protected Feature[] featureSet;



		/**
		 * Array of OutputClass objects, that represents the possible output classes of the problem
		 */
		protected OutputClass[] outputClasses;	

//		==Data members

		/**number of classes*/
		protected int numClasses;

		/**number of categories for categorical features and number of bins for numerical, then discretized, features . Each position refers to
		 * a different feature*/
		protected Integer[] numValues;


		protected int numFeats;
		protected int trainInstances;
		protected static int testInstances;
		protected double[][] probabilities;

		//static members, needed to track the overall number of instances 
		//across different runs of CV
		protected static int cvInstances;
		protected  Integer currentExp;

//		the following variables store prior counts to be added to the empirical frequencies.
//		the counts are designed to implement a uniform prior.
//		note that all prior counts variables are named pcType.
//		this is 1/nc (nc=number of classes)

		/**prior counts for classes*/
		protected double pcClass; 

		/**prior counts for conditional frequencies*/
		protected double[] pcCond;

		/**prior counts for unconditional frequencies*/
		protected double[] pcUncond;
		protected int s;
		int[][] confMatrix;



		//==NESTED CLASS Feature
		/**
		 * Helper class for Naive Classifiers, that implements Mar and NonMar features.
		 * Features are characterized by the bivariate counts of their effective occurrences 
		 * (Frequencies), by the number of missing data for each output class (Missing) and
		 * by  the logarithm of conditioned probabilities (LogProbability)
		 */
		protected  class Feature implements Serializable{

                       /** for serialization */
                       static final long serialVersionUID = -1478242251770381214L;

			/**Constructor that copies the name and the conditionalFreq table, and computes the log-probabilities table
			 * 		 */
			Feature (String SuppliedName, double[][] SuppliedFrequencies, int[] SuppliedMissing) 
			{
				name=SuppliedName;
				conditionalFreq=SuppliedFrequencies.clone();
				missing=SuppliedMissing.clone();
				uncondFrequencies=new double[conditionalFreq[0].length];

				int i,j;
				for (i=0; i<conditionalFreq[0].length; i++){
					uncondFrequencies[i]=0;
					for (j=0; j<conditionalFreq.length;j++)
						uncondFrequencies[i]+=conditionalFreq[j][i];
				}

				//calculate log of probabilities
				logProbability=new double[conditionalFreq.length][conditionalFreq[1].length];
				double  currentCountSum;


				for (i=0; i<conditionalFreq.length; i++)
				{
					currentCountSum=ArrayUtils.arraySum(conditionalFreq[i])[0];
					for (j=0; j<conditionalFreq[1].length; j++)
					{
						logProbability[i][j]=Math.log((double)conditionalFreq[i][j]/currentCountSum);
					}
				}
			}



			/**Counts that correspond to counts-after-dropping-missing for MarFeatures,
			 * bivariate count: frequency are computed for each output class and for each class of the
			 * feature. They are double to manage possible partial units due to the prior. The rows refer to the different output classes, and the columns to the 
			 * different feature classes.
			 */
			private final double[][] conditionalFreq;

			/**Simple uncondFrequencies, not conditioned. Useful to computed Bma*/
			private final double[] uncondFrequencies;

			/**Logarithm of conditioned probabilities: Log(P(ai|c)) 
			 */
			private final double[][] logProbability;

			/**How many times the feature is missing, for every output class.
			 */
			private final int[] missing;

			/**Name
			 * 
			 */
			private final String name;

			/**
			 * @return Returns the conditionalFreq.
			 */
			public double[][] getConditionalFreq() {
				return conditionalFreq;
			}

			/**
			 * @return Returns the conditionalFreq for a specified class
			 */
			public double[] getCondFrequencies(int ClassIdx) {
				return conditionalFreq[ClassIdx];
			}


			/**
			 * @return Returns the conditionalFreq for a specified class
			 */
			public double[] getUncondFrequencies() {
				return uncondFrequencies;
			}


			/**
			 * @return Returns the conditionalFreq for a specified class, computed as
			 *counts of those records where the class is the one required and the value of the given MAR feature
			 *is not missing
			 */
			double getClassCountAsMar(int ClassIdx) 
			{
				double[] tmpArr= ArrayUtils.arraySum(conditionalFreq[ClassIdx]);
				return tmpArr[0];
			}


			/**
			 * @return Returns the conditionalFreq for a specified class, and for 
			 * a specified class (range of values) defined within the feature domain
			 */
			double  getConditionalFrequencies(int ClassIdx, int FeatureClassIdx) {
				return conditionalFreq[ClassIdx][FeatureClassIdx];
			}




			/**
			 * @return Returns the log of cond probabilities for a specified class
			 */
			double[] getLogProbability(int ClassIdx) {
				return logProbability[ClassIdx];
			}

			/**
			 * @return Returns the log of cond probabilities for a specified class and
			 * for a specific value of the feature
			 */
			double getLogProbability(int ClassIdx, int FeatureValue) {
				return logProbability[ClassIdx][FeatureValue];
			}

			/**
			 * @return Returns the whole log of cond probability table 
			 */
			double[][] getLogProbability() {
				return logProbability;
			}

			/**
			 * @return Returns the missing.
			 */
			int[] getMissing() {
				return missing;
			}

			/**
			 * @return Returns the number of missing data
			 * for a given output class
			 */
			int getMissing(int OutputClass) {
				return missing[OutputClass];
			}

			/**
			 * @return Returns the name.
			 */
			String getName() {
				return name;
			}

		}
		//==END NESTED CLASS Feature



		//==NESTED CLASS: OutputClass
		/**
		 * Helper class for Naive Classifiers, that implements the output class of the classification problem.
		 */
		protected class OutputClass implements Serializable{
                      /** for serialization */
                        static final long serialVersionUID = -1478242251770381214L;


			//constructor 
			OutputClass (String SuppliedClassName, double SuppliedClassFrequency, double suppliedLogProbability) 
			{
				name=SuppliedClassName;
				frequency=SuppliedClassFrequency;
				logProbability=suppliedLogProbability;
			}

			double getFrequency() {
				return frequency;
			}
			String getName() {
				return name;
			}

			double getLogProbability()
			{
				return logProbability;
			}

			double frequency;

			/**Stores log-probability of each cless
			 */
			double logProbability;


			/**names of the output classes*/
			String name;

		}

}


