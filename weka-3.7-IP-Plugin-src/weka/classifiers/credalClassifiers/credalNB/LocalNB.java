/**
 * LocalClassifier.java
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

import java.util.ArrayList;
/**Local Naive Bayes
 * 
 */
public class LocalNB extends LocalNaive{	

      /** for serialization */
      static final long serialVersionUID = -1478242251770381214L;

	LocalNB(String distancePar, boolean weightedPar, int numExperiments, int prior, int nClasses){
		super(distancePar, weightedPar);
		numClasses=nClasses;
		confMatrix=new int[numClasses][numClasses];
		accuracy= new double[numExperiments];
		priorType=prior;		
	}


	/**Set the counts arising only from then prior distribution
	 * over contTables and classFreqs, thus removing 
	 *any count arising from previous local data sets.
	 */
	private void addPseudoCount(){
		int i,j,k;
		psContTables=new double[numFeats][][];
		for (i=0;i<numFeats;i++){
			psContTables[i]=new double[numValues[i]][numClasses];
		}
		psClassFreqs=new double [numClasses];		

		for (i=0;i<numFeats;i++){
			for (j=0;j<numValues[i];j++){
				for (k=0;k<numClasses;k++){
					psContTables[i][j][k]=contTables.get(bwdthIdx)[i][j][k];
				}
			}
		}

		for (k=0;k<numClasses;k++){
			psClassFreqs[k]=classFreqs.get(bwdthIdx)[k];
		}

		pcFeats=new double[numFeats];
		//prior counts for classes and conditional frequencies
		if (priorType==1){
			pcClass=s;
			for (i=0; i<numFeats; i++){
				pcFeats[i]=s; 
			}
		}

		else if (priorType==2){
			pcClass=(double)s/numClasses; 

			for (i=0; i<numFeats; i++){
				pcFeats[i]=(double)s/(numClasses*numValues[i]); 
			}
		}

		else{
			System.out.print("Wrong prior specified");
			System.exit(0);
		}



		//add priors
		for (i=0;i<numClasses;i++){
			psClassFreqs[i]+=pcClass;
		}

		for (i=0;i<numFeats;i++){
			for (j=0;j<numValues[i];j++){
				for (k=0;k<numClasses;k++){
					psContTables[i][j][k]+=pcFeats[i];
				}
			}	
		}
	}


	/**
	 * Setup non static vars
	 */
	public void setup (ArrayList<int[]> testingSet){
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
		predictions = new int[testingSet.size()];
		probabilities = new double [testingSet.size()][numClasses];
		logProbs=new double[numClasses];
		//we'll be updated to 0 by setCurrentInstance before classifying the first instance 
		instanceIdx=-1;
		currentExp++;
	}

	/**
	 * Classify the current instance; if weigthed, it populates the contingency tables in a weighted way, otherwise
	 * if just copies the provided tables; the prediction is saved in predictions[i] while the probability distribution is saved 
	 * in probabilities[i][].
	 */
	
	public void classifyCurrentInst(ArrayList<double[][][]> contTablesPar,ArrayList<double[]> classFreqsPar){

//==this commented code needed in case LNBC is not weighted		
//		if (distWeighted){
//			ranker.computeWeights();
//			resetTables();
//			updateTablesW();
//		}
//		else{
			contTables=contTablesPar;
			classFreqs=classFreqsPar;
//		}
		addPseudoCount();
		computeProbs(currentInst);
		if (instanceIdx==testInstances-1){
			accuracy[currentExp]/=testInstances;
		}
	}


	/**
	 * Computes the actual probabilities of the different classes;
	 * stores the index of the class with the highest log-prob into predictions
	 * and the probability distributions in probabilities[i][].
	 * Called by function classifyInstances.
	 */
	private void computeProbs (int[] instance){
		int i,j,ii;
		double maxLogProb=-Double.MAX_VALUE;
		int maxProbIdx=-1;
		double sum;

		//compute log-probabilities.
		//remember to consider the prior
		for (j=0;j<numClasses;j++){
			logProbs[j]=Math.log(psClassFreqs[j]/(stop_inst+1));
			for (i=0;i<numFeats;i++){
				if (instance[i]==-9999){
					continue;
				}
				//as the data set can contain missing data, we do not have to divide for number 
				//of instances having class j, but rather for the number of instances which have class j AND
				//in which feature i is not missing.
				sum=0;
				for (ii=0;ii<numValues[i];ii++){
					sum += psContTables[i][ii][j];
				}
				logProbs[j] += Math.log(psContTables[i][instance[i]][j]) -Math.log(sum);
			}
			if (logProbs[j]>maxLogProb){
				maxLogProb=logProbs[j];
				maxProbIdx=j;
			}
		}

		//now we compute the actual probabilities and we store the results
		//in data members predictions[] and probabilities[][].
		//we adopt a numerically robust approach. 
		double shift=logProbs[maxProbIdx];
		double sumProb=0;
		double[] tmpArr = new double [numClasses];

		for (i=0; i<numClasses; i++)	{
			tmpArr[i]=Math.exp(logProbs[i]-shift);
			sumProb+=tmpArr[i];
		}

		//store the results
		for (i=0; i<numClasses; i++){
			probabilities[instanceIdx][i]=tmpArr[i]/sumProb;
		}
		predictions[instanceIdx]=maxProbIdx;

		//update confusionMatrix
		confMatrix[instance[numFeats]][maxProbIdx]++;

		//update accuracy vector and nbcAccurate if accurate
		if (maxProbIdx==instance[numFeats]){
			accuracy[currentExp]++;
		}
	}


	public  double[] getAccuracy() {
		return accuracy;
	}

	public  int[] getPredictions() {
		return predictions;
	}

	/**1 means laplace, 2 perks*/
	private int priorType;

	/**prior counts for classes*/
	private double pcClass; 

	/**
	 * pseudo contingency tables
	 */
	private double[][][] psContTables;
	private double[] psClassFreqs;

	/**
	 * Probabilities estimated for each class, for each instance
	 */
	private double[][] probabilities;
	private int[] predictions;

	/**prior counts for features*/
	private double[] pcFeats;
	private double[] logProbs; 
	private double[] accuracy;
}
