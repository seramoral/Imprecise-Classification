package weka.classifiers.credalClassifiers.credalNB;

import java.util.ArrayList;

public class FeatureSelector {
	FeatureSelector (ArrayList<int[]> suppliedTrSet,  ArrayList<String> featureNames, ArrayList<String>classNames, ArrayList<Integer> numClassForEachFeature)
	{
		trainingSet=suppliedTrSet;
		//we assume numExperiments to be 1, priorCode to be 2 (Perks) and map to be true.
		nbc=new NaiveBayes(1,classNames.size());
		//we force s=8 (less aggressive with informative features) and global prior
		nbc.train(trainingSet, featureNames, classNames, numClassForEachFeature, 2, true,8);
		processFeatMap(nbc.getMapArchitecture());
		
	}
	
	/**Generates selectedFeats (array of indexes of selected features) starting from an
	 * array containing +1 for selected features and -1 for not selected features.
	 */
	private void processFeatMap(int[] featureMap){
		selectedFeats=new ArrayList<Integer>(featureMap.length);
		int i;
		for (i=0; i<featureMap.length;i++){
			if (featureMap[i]==1){
				selectedFeats.add(i);
			}
		}
	}
	
	public ArrayList<Integer> getSelectedFeats(){
		return selectedFeats;
	}
	
	//DATA MEMBERS
	private ArrayList<int[]> trainingSet;
	private ArrayList<Integer> selectedFeats;
	private NaiveBayes nbc;
}






