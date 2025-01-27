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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

/**Abstract super-class for Naive Classifiers 
 * 
 */
public  abstract class LocalNaive implements Serializable{

      /** for serialization */
      static final long serialVersionUID = -1478242251770381214L;

	/**
	 * It initializes a few data members.
	 */
	LocalNaive(String distancePar, boolean weightedPar)
	{	
		distType=distancePar;
		distWeighted=weightedPar;
		if (currentExp==null){
			currentExp=-1;
		}
		s=1;
		bwdthIdx=0;
		stepsize=20;
	}

	/**
	 * Initializes only distWeighted.
	 */
	LocalNaive(boolean weightedPar)
	{
		distWeighted=weightedPar;
	}


	protected  void fillContTable (){
		
		start_inst=0;
		stop_inst=k;
		resetTables();

		ranker.computeRank(currentInst);
		if (distWeighted){
			ranker.computeWeights();
			updateTablesW();
		}
		else{
			updateTablesUnw();
		}
	}
	
	
	/**Set to 0 all frequencies.
	 * @return 
	 */
	protected  void resetTables(){

		int i,j;

		//reset classFreqs
		Arrays.fill(classFreqs.get(bwdthIdx), 0);

		for (i=0;i<numFeats;i++){
			for (j=0;j<numValues[i];j++){
				Arrays.fill(contTables.get(bwdthIdx)[i][j], 0);
			}
		}	
	}

	//abstract void classifyCurrentInst();


	/**
	 *Allocates classFreqs and contTables
	 */
	protected void setup(ArrayList<int[]> passedTrSet, Integer[] infoArray){
		
	}
	
	/**Setup of static variables of LocalClassifier, i.e., trSet, numFeats, numValues,k, ranker.
	 */
	protected  void setupStaticVars(ArrayList<int[]> passedTrSet, Integer[] infoArray){

		trSet=passedTrSet;
		numFeats=infoArray[0];
		numValues=new Integer[numFeats];
		System.arraycopy(infoArray, 1, numValues, 0, numFeats);
		k=25;
		if (trSet.size()<k){
			k=trSet.size();
		}
		int i;
		orderIndexes= new Triple[trSet.size()];

		for (i=0;i<orderIndexes.length;i++){
			orderIndexes[i]=new Triple(i);
		}
		ranker = new Ranker();
	}





	/**Add unweighted empirical counts to contTables
	and to classFreqs; the range of instances involved are from start_instance and stop_instance*/
	 void updateTablesUnw(){
		int i,j;
		int[] currentInstance;
		int currentClass;
		for (i=start_inst;i<stop_inst;i++){
			currentInstance=trSet.get(orderIndexes[i].getIndex());
			currentClass=currentInstance[numFeats];
			classFreqs.get(bwdthIdx)[currentClass]++;
			for (j=0;j<numFeats;j++){
				if (currentInstance[j] !=-9999){
					contTables.get(bwdthIdx)[j][currentInstance[j]][currentClass]++;
				}
			}
		}
	}

	/**Add weighted empirical counts to the contingency tables,
	which should contain only the prior counts;
	get counts on the nearest k instances*/
	 void updateTablesW(){
		int i,j;
		int[] currentInstance;
		int currentClass;
		double currentWeigth;
		for (i=0;i<stop_inst;i++){
			currentInstance=trSet.get(orderIndexes[i].getIndex());
			currentWeigth=orderIndexes[i].getWeight();
			currentClass=currentInstance[numFeats];
			classFreqs.get(bwdthIdx)[currentClass] += currentWeigth;
			for (j=0;j<numFeats;j++){
				if (currentInstance[j] !=-9999){
					contTables.get(bwdthIdx)[j][currentInstance[j]][currentClass] += currentWeigth;
				}
			}
		}
	};
	
	/**Set the current instance, rank all the instances of the training set and fills accordingly 
	 * the contTables, using k=100.
	 * @param currentInst
	 */
	public static void setCurrentInst(int[] currentInst) {
		LocalNaive.currentInst = currentInst;
		instanceIdx++;
		Arrays.sort (orderIndexes, idxComparator);
	}
	
	//==GETTERS AND SETTERS
	public ArrayList<double[][][]> getContTables(){
		return contTables;
	}
	
	public ArrayList<double[]> getClassFreqs(){
		return classFreqs;
	}


	//==DATA MEMBERS
	/**number of classes*/
	protected static int numClasses;

	/**number of categories for categorical features and number of bins for numerical, then discretized, features . Each position refers to
	 * a different feature*/
	protected static Integer[] numValues;
	protected static double s;

	protected static int testInstances;

	/**number of features*/
	protected static int numFeats;

	protected static Integer currentExp;
	protected static int[] currentInst;

	 
		
	
	
	
	//which bandwidth we are currently working with
	protected static int bwdthIdx;
	
	protected static IndexComparator idxComparator;

	/**needed as local classifiers are memory-based */
	protected static ArrayList<int[]> trSet;

	/**contains the empirical frequencies on the first k instances, without any pseudo-count added;
	 * the counts might span more than k instances in case the distance of the (k+1) instance equals
	 * that of the (k-th) instance; the first dimension indexes the bandwidth we are referring to 
	 * (k, k+stepsize, k+2*stepsize ecc).
	 * 
	 */
	protected ArrayList<double[][][]> contTables;
	protected ArrayList<double[]> classFreqs;
	protected static int instanceIdx;

	/**
	 *For each query, this will contain the distance and the index of each instance.
	 *It will be sorted, so that to learn the classifier it is enough to consider the first k instances
	 * according to the provided order.
	 * It is static, so any local classifier can access the ordered indexes.
	 */
	protected static Triple[] orderIndexes;

	/**how much we increase the bandwidth between different local classifiers*/
	protected static int stepsize;

	/**inherited by all subclasses*/
	protected int[][] confMatrix;

	/**tool for comparing the Triple objects*/
	//	protected static Triple tripleComparer;

	/**number of neighbors to be considered*/
	protected static int k;

	/**when classifying a certain instances, the number of used neighbors can be slightly different from k, 
	 * but on the next instance we will restore the original k */
	protected static int stop_inst;
	protected static int start_inst;
	protected static Ranker ranker;

	/** whether we weight or not on the base of the distance; this is not a static field
	 * because the credal and the bayesian might be one weighted and the other unweigthed*/
	protected boolean distWeighted;
	protected static String distType;

	/**
	 * Helper class used to order the instances of the training set according to the 
	 * distance from the query point
	 */
	class Ranker{
		Ranker(){
			if (distType.equalsIgnoreCase("mvdm")){
				initContingencyTable();
				compMvdmContTable();
				compMvdmDistanceTable();
			}
		}

		/**
		 * Computes the distances of the training instances from the query; stores the results
		 * in orderIndexes and sort it. After the function has ended, orderIndexes contain the index of the instances
		 * sorted from the closes to the nearest. The bandwidth k is then extended, to contain all the instances that are at the same distance
		 * of the k-th from the query (if any); so that the bandwidth includes all instances having distance <= d_k from the query.
		 * If necessary, the weights of each instance (inversely proportional 
		 * to the distance) are also computed.
		 */
		public void computeRank(int[] query){

			int i; int[] currentInstance;

			//this implementation duplicate a few rows of code, but avoid to 
			//perform many unnecessary if.
			if (distType.equalsIgnoreCase("mvdm"))
			{
				for (i=0; i<trSet.size();i++){
					currentInstance=trSet.get(i);
					orderIndexes[i].setDistance(mvdmDistance(currentInstance, query));
				}
			}
			else if (distType.equalsIgnoreCase("overlap"))
			{
				for (i=0; i<trSet.size();i++){
					currentInstance=trSet.get(i);
					orderIndexes[i].setDistance(overlapDistance(currentInstance, query));
				}
			}
			else{
				System.out.println("Bad distance type");
				System.exit(0);
			}


			//this will orderIndexes according to the distance.
			Arrays.sort(orderIndexes);
			setStopInstance();

			//old code
//			if (distWeighted){
//				computeWeights();
//			}

		}


		/**Sets the value of stop_instance, by:<p>
		 * 1)looks for the distance of the k-th instance; if there are further instances (k+1, k+2 etc) at the same distance
		 * of the k-th, then the value of k is extended to include all of them. This way, we consider all the instances
		 * which have distance <= d_k. 
		 * 2)In case all the first k instances have distance 0, extends the bandwidth to contain at least with one instance
		 * with distance >0
		 * REMARK: the modified bandwidth is written in tmp_k, so that the original value is kept unchanged in k.
		 * Before updating the bandwidth therefore, tmp_k is set to the original bandwidth k. 
		 * For most instances however the bandwidth will correspond to the actual user set value of k; only if the k-th
		 * distance equals the k+1 (or further others), the number of neighbors will be changed.
		 */
		private void setStopInstance(){
			while ( (stop_inst<orderIndexes.length-1) &&
					(orderIndexes[stop_inst-1].getDistance().compareTo(orderIndexes[stop_inst].getDistance())==0)) 
			{
				stop_inst++;
			}


			//case bandwidth 0
			//we extend k to include at least one element with bandwidth>0
			if (orderIndexes[stop_inst-1].getDistance()==0){
				stop_inst++;
				//the added element has distance >0
				//let's check whether there are other instances at the same distance
				while ( (stop_inst<orderIndexes.length) &&
						(orderIndexes[stop_inst-1].getDistance().compareTo(orderIndexes[stop_inst].getDistance())==0)) 
				{
					stop_inst++;
				}
			}
		}

		/**
		 * Reads from orderIndexes the distance of each instance from the query, and
		 * computes accordingly the weight of each instance, which decreases linearly with the distance.
		 * The weight given to the (k+1) instance is 0; moreover, the weights are finally rescaled to sum
		 * to k. (see the paper by Eibe Frank on locally weighted naive bayes.) 
		 */
		public void computeWeights(){
			int i;
			double maxDist=orderIndexes[stop_inst-1].getDistance();

			//mvdm computes distance=0 if all features are missing
			if (maxDist==0){
				maxDist=Double.MIN_NORMAL;
			}
			double sumWeight=0;
			double tmp;

			//the algorithm is borrowed from WEKA's implementation. Basically, w=max (1.001-dist[i]/maxDist,0).
			//however, we compute the weights only for the instances whose distance is <= bandwidth and therefore 
			//all of them will have positive weight. Note that the instance with distance=d_k will be given weight 0.001.

			for (i=0;i<stop_inst;i++){
				orderIndexes[i].setWeight(tmp=1.0001-(orderIndexes[i].getDistance()/maxDist));
				if (orderIndexes[i].getWeight()<0){
					orderIndexes[i].setWeight(0);
				}
				sumWeight += tmp;
			}

			//rescale the weigths to have sum k
			for (i=0;i<stop_inst;i++){
				orderIndexes[i].setWeight(orderIndexes[i].getWeight()*stop_inst/sumWeight);
			}
		}





		/**
		 * Initializes the contingency tables according to the composite prior.
		 * The computed distance will be almost unchanged, but this allows to deal with the case in which 
		 *  a given value of a certain feature is totally absent from the training set (possible, as we are delaing with
		 *  local models).
		 */
		private void initContingencyTable(){
			int i;
			mvdmContTable=new double[numFeats][][];
			for (i=0; i<numFeats;i++){
				mvdmContTable[i]=new double[numValues[i]][numClasses+1];
			}
		}


		/**
		 * Scans the whole dataset and fills the globalContTable; the last element of each row of each table
		 * contains the sum of the frequencies, for a fixed value of the feature, over all the classes.
		 */
		private void compMvdmContTable(){
			int i,j;

			int currentValue;
			int[] currentInstance =new int[numFeats];
			int currentClass;
			for (i=0;i<trSet.size();i++){
				currentInstance=trSet.get(i);
				currentClass=currentInstance[numFeats];
				for (j=0;j<numFeats;j++){
					currentValue=currentInstance[j];
					if (currentValue != -9999){
						mvdmContTable[j][currentValue][currentClass]++;
						//update the last column, which stores the total for 
						//a given value of the feature 
						mvdmContTable[j][currentValue][numClasses]++;
					}
				}
			}
		}


		private void compMvdmDistanceTable(){

			mvdmTable= new double[numFeats][][];
			avgMvdm=new double[numFeats];
			int i,j,k;
			int counter;
			double sum;
			for (i=0;i<numFeats;i++){
				mvdmTable[i]=new double [numValues[i]][numValues[i]];
			}

			//now we fill the upper half of the matrix
			//the diagonal remains untouched, as it will be 0 even doing the actual computation.

			for (i=0;i<numFeats;i++){
				counter=0; sum=0;
				for (j=0;j<numValues[i];j++){
					for (k=j+1;k<numValues[i];k++){
						mvdmTable[i][j][k]=mvdmSingleDistance(i,j,k);
						//the matrix has to be symmetrical.
						//a different implementation (triangular matrix) would save memory 
						//but perhaps slower, as to access it we would need to sort j and k before.
						mvdmTable[i][k][j]=mvdmTable[i][j][k];
						sum+=mvdmTable[i][k][j];
						counter++;
					}
				}
				//mean expected probability
				avgMvdm[i]=sum/counter;
				sum=0; counter=0;
				for (j=0;j<numValues[i];j++){
					for (k=j+1;k<numValues[i];k++){						
						sum+=Math.abs(mvdmTable[i][k][j]-avgMvdm[i]);
						counter++;
					}
				}
				//average distance of the average value
				avgMvdm[i]=sum/counter;
			}

		}


		/**
		 * Computes mvdm distance between two values of a single feature.
		 */
		private double mvdmSingleDistance(int featIdx, int from, int to){
			int i;
			double distance =0;
			for (i=0;i<numClasses;i++){
				distance += Math.abs(mvdmContTable[featIdx][from][i]/mvdmContTable[featIdx][from][numClasses]
				                                                                                  -mvdmContTable[featIdx][to][i]/mvdmContTable[featIdx][to][numClasses]);
			}			
			return distance;
		}
		/**
		 *If instance[i] or query[i] is missing, then the i-th component of the distance is 0, temporarily. 
		 */
		private double mvdmDistance (int[] instance, int[] query){
			double distance=0;
			int i;
			for (i=0;i<numFeats;i++){
				//for the moment, having a missing coefficient either in the query or in the
				//instances returns 0, but in future we could elaborate a bit better
				if (instance[i]==-9999 | query[i]==-9999){
					distance += avgMvdm[i]; }
				else{
					distance += mvdmSingleDistance(i,instance[i],query[i]);
				}
			}
			return distance;
		}

		/**
		 * Euclidean overlap distance, i.e., we add 1 for each feature which mismatch the velu in tue query and eventually 
		 * we apply the sqrt.
		 */
		private double overlapDistance (int[] instance, int[] query){
			double distance=0;
			int i;
			for (i=0;i<numFeats;i++){
				//implemented as in WEKA: if any value is missing, 
				//we add 1 to the distance. Not very smart though.
				if ((instance[i]==-9999) | (query[i]==-9999)){
					distance++;
				}

				else if (! (instance[i]==query[i])){
					distance++;
				}
			}
			return Math.sqrt(distance);
		}



		/**tables for mvdm distance. They will contain symmetrical values.  */
		private double [][][] mvdmTable;

		/**the average distance for each feature*/
		private double[] avgMvdm;

		/**for each feature, a table; on the vertical the re are the feature values, on the 
		 * horizontal the class values; th last column is the sum over all the class, for a given feature
		 * value.
		 */

		private double [][][] mvdmContTable;
	}

	/**Helper class which stores the order of the instances and, if needed, their weights*/
	class Triple implements Comparable<Object> {

		//only sets the index
		public Triple(int indexPar){
			index=indexPar;
		}

		public Double getDistance() {
			return distance;
		}

		public int compareTo (Object anotherTriple) throws ClassCastException {
			if (!(anotherTriple instanceof Triple))
				throw new ClassCastException("A Triple object expected.");
			return this.distance.compareTo(((Triple)anotherTriple).getDistance());
		}

		public void setDistance(double distance) {
			this.distance = distance;
		}

		public double getWeight() {
			return weight;
		}

		public void setWeight(double weight) {
			this.weight = weight;
		}

		public Integer getIndex() {
			return index;
		}

		Double distance;
		Integer index;
		double weight;

	}

	public  class IndexComparator implements Comparator<Object> {
		public int compare(Object triple, Object anotherTriple) {
			Integer index1= ((Triple)triple).getIndex();
			Integer index2= ((Triple)anotherTriple).getIndex();
			return index1.compareTo(index2);
		}
	}


}


