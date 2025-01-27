/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

package weka.classifiers.credalClassifiers;

import weka.classifiers.trees.*;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Capabilities.Capability;
import weka.classifiers.*;
import weka.core.*;
import java.util.*;
import static weka.classifiers.credalClassifiers.CredalDecisionTree_CostErrors_Weighted.IMPRECISE_ENTROPY;

/**
 *
 * @author Serafin
 */
public class CredalDecisionTree_CostErrors_Weighted extends CostSensitiveCredalClassifier implements OptionHandler, AdditionalMeasureProducer{
    /* Field that contains the tree structure*/
    NodeTree m_RootNode;
    
    /* Level of the tree when the building of tree stops altought there is an improvement in the entropy. 
     If it sets to 0, the unique stop criterium is the criterium of deterioration of the entropy*/
    int m_StopLevel=0; 
  
    /** Constant values that indicate the entropy that it is used to build the decision tree.*/
    public static final int IMPRECISE_ENTROPY=0;
   
    /** This field stores the index of the entropy criterium used to build the decision tree*/
    int m_SplitMetric=IMPRECISE_ENTROPY;
  
  
    static final String[] STRING_SPLIT_METRIC={"A_NPI_M"};
    public static final Tag[] TAGS_SPLIT_METRIC = { new Tag(IMPRECISE_ENTROPY, STRING_SPLIT_METRIC[0]), };
    
     /* if this field is set to false, the class of a leave where the number of occurrences of the configuration
        that defines is zero is set to the most probable class that defines its parent if if was a leave */
    boolean m_MisclassifiedAllowed=true;
  
    /* Vector with the ingoGain value of each attribute */
    double[] infoGains;
    
  
  
    /* If this field contains a value k then the attribute with k-th hightest value of Information Gain will be used
      as a root of the node. The default value is k=1.*/
    int m_KThRootAttribute=1;
    
    /* Returns an instance of a TechnicalInformation object, containing 
    * detailed information about the technical background of this class,
    * e.g., paper reference or book this class is based on.
    * 
    * @return the technical information about this class
    */
    
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation   result;
    
        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Joaquin Abellan, and Andres Masegosa");
        result.setValue(Field.YEAR, "2012");
        result.setValue(Field.TITLE, "Imprecise Classiffication with Credal Decision Trees");
        result.setValue(Field.JOURNAL, "International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems");
        result.setValue(Field.VOLUME, "20");
        result.setValue(Field.NUMBER, "5");
        result.setValue(Field.PAGES, "763-787");

        return result;
    }
    
    
    /** Returns default capabilities of the classifier.
    *
    * @return      the capabilities of this classifier
    */
    @Override
  
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);
    
        return result;
    }
    
    
    /** Returns a superconcise version of the model
    @return a summary of the model
    */
  
    public String toSummaryString() {

        return "Number of leaves: " + numLeaves(m_RootNode) + "\n"
            + "Size of the tree: " + numNodes(m_RootNode) + "\n";
    }
   
    /* Returns number of leaves in tree structure.
    
    */
  
    public double numLeaves(NodeTree node) {
        double num = 0;
        int i;
    
        if (node.getAttribute()==null)
            return 1;
        
        else
            for (i=0;i<node.getAttribute().numValues();i++)
                num += numLeaves(node.getSuccesors(i));
        
        return num;
    }

    /* Returns number of nodes in tree structure.
    */
    public double numNodes(NodeTree node) {
        double no = 1;
        int i;
    
        if (node.getAttribute()!=null){
            no=1; 
       
            for (i=0;i<node.getAttribute().numValues();i++)
                no += numNodes(node.getSuccesors(i));
        }
    
        return no;
    }

    /* Returns the size of the tree
    * @return the size of the tree
    */
    public double measureTreeSize() {
        return numNodes(this.m_RootNode);
    }

    /* Returns the number of leaves
    * @return the number of leaves
    */
    public double measureNumLeaves() {
        return numLeaves(this.m_RootNode);
    }

    /* Returns the number of rules (same as number of leaves)
    * @return the number of rules
    */
  
    public double measureNumRules(){
        return numLeaves(this.m_RootNode);
    }
    
    /* Returns an enumeration of the additional measure names
    * @return an enumeration of the measure names
    */
  
    public Enumeration enumerateMeasures() {
        Vector newVector = new Vector(3);
        newVector.addElement("measureTreeSize");
        newVector.addElement("measureNumLeaves");
        newVector.addElement("measureNumRules");

        Enumeration enume = super.enumerateMeasures();
        
        while(enume.hasMoreElements())
            newVector.add(enume.nextElement());

        return newVector.elements();
    }
    
      /* Returns the value of the named measure
   * @param measureName the name of the measure to query for its value
   * @return the value of the named measure
   * @exception IllegalArgumentException if the named measure is not supported
   */
  
    public double getMeasure(String additionalMeasureName) {
        if (additionalMeasureName.compareToIgnoreCase("measureNumRules") == 0) {
            return measureNumRules();
        }
        
        else if (additionalMeasureName.compareToIgnoreCase("measureTreeSize") == 0) {
            return measureTreeSize();
        }
        
        else if (additionalMeasureName.compareToIgnoreCase("measureNumLeaves") == 0) {
            return measureNumLeaves();
        }

        return super.getMeasure(additionalMeasureName);
    }
    
        /* Returns a string describing the classifier.
    * @return a description suitable for the GUI.
    */
    
    public String globalInfo() {

        String output = " Method based on buiding a Credal Decision Tree for Imprecise Classification with the A-NPI-M"
                + " weighting the instances by taking into account the cost of errors of each class value as well as"
                + " the distribution of relative frequencies that reaches the maximum of entropy with the A-NPI-M"
            + "\n\nFor more information, see\n\n"
            + getTechnicalInformation().toString()
            + "\n\nOptions:\n\n";

        Enumeration enu = this.listOptions();
        while (enu.hasMoreElements()) {
            Option option = (Option) enu.nextElement();
            output=output.concat(option.synopsis() + ' ');
            output=output.concat(option.description() + "\n");
        }

        return output;
    }
    
            /**
    * Gets the attribute used for splitting in a certain node
    * @param data the instances of the node
    * @param level the level of the node
    * @return the split attribute
    * @throws Exception 
    */
  
    protected Attribute getAttributeToRamify(Instances data, int level) throws Exception{
    
        // Compute attribute with maximum information gain.
        infoGains = new double[data.numAttributes()];
        Enumeration attEnum = data.enumerateAttributes();
        
        //Compute the information gain for each attribute
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            infoGains[att.index()] = computeInfoGain(data, att);
        }
        
        infoGains[data.classIndex()]=-Double.MAX_VALUE;
    
        if (level==0)
            return data.attribute(Utils.sort(infoGains)[infoGains.length-this.m_KThRootAttribute]);
        
        else
            return data.attribute(Utils.maxIndex(infoGains));
    }
    
  
    
    
      /* Builds the Imprecise Credal Decision tree classifier with cost errors.
    *
    * @param data the training data
    * @exception Exception if classifier can't be built successfully
    */
    
    @Override
    public void buildClassifier(Instances data) throws Exception {

        initStats();
        
        // can classifier handle the data?
        getCapabilities().testWithFail(data);
        
        computeMatrixCostErrors(data);
        computeClassWeights(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
    
        this.m_RootNode=new NodeTree();
    
        makeTree(this.m_RootNode,data,0);
    }
    
       /* Computes class distribution for instance using decision tree.
    *
    * @param instance the instance for which distribution is to be computed
    * @return the class distribution for the given instance
    * @throws NoSupportForMissingValuesException if instance has missing values
    */
    
    public double[] distributionForInstance(Instance instance) throws Exception {

        if (instance.hasMissingValue()) 
            throw new NoSupportForMissingValuesException("IPTree: no missing values, " + "please.");
    

        this.updateCredalStatistics(instance);

        double [] probs = new double[instance.numClasses()];
        
        for (int i=0; i<probs.length; i++)
            probs[i]=Double.NaN;

        return probs;
    }
    
    /* Computes class distribution for instance using decision tree.
    *
    * @param instance the instance for which distribution is to be computed
    * @return the class distribution for the given instance
    * @throws NoSupportForMissingValuesException if instance has missing values
    */
  
    public double[] frequencyForInstance(Instance instance) throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("IPTree: no missing values, "    + "please.");
        }
    
        return this.frequencyForInstance(this.m_RootNode, instance);

    }
    
      /* Computes class distribution for instance in a node of the tree.
    * @param node the node in which distribution is computed
    * @param instance the instance for which distribution is to be computed
    * @return the class distribution for the given instance
    * @throws NoSupportForMissingValuesException if instance has missing values
    */

    private double[] frequencyForInstance(NodeTree node, Instance instance){
      
        if (node.getAttribute()== null) {
            return node.getFrequency();
        }
        
        else { 
            return frequencyForInstance(node.getSuccesors((int) instance.value(node.getAttribute())), instance);
        }
      
    }
    
     /* Classifies a given test instance using the decision tree.
    * @param instance the instance to be classified
    * @return the classification
    * @throws NoSupportForMissingValuesException if instance has missing values
    */
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("IPTree: no missing values, " + "please.");
        }

        this.updateCredalStatistics(instance);

        return Double.NaN;
    }
    
      /* Splits a dataset according to the values of a nominal attribute.
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
    */
   
    protected Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
    
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }
        
        Enumeration instEnum = data.enumerateInstances();
        
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            splitData[(int) inst.value(att)].add(inst);
        }
    
        for (int i = 0; i < splitData.length; i++) {
            splitData[i].compactify();
        }
        
        return splitData;
    }
    
     /**
    * Compute the class distribution of a given node
    * @param node the node for which the distribution is computed
    * @param data the instances of the node
    */
    
    protected void computeClassDistribution(NodeTree node, Instances data){
        //node.setDistribution(new double[data.numClasses()]);
        node.setFrequency(new double[data.numClasses()]);
        Enumeration instEnum = data.enumerateInstances();
      
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            //node.getDistribution()[(int) inst.classValue()]++;
            node.getFrequency()[(int) inst.classValue()]++;
        }
      //node.setSupport(data.numInstances());

    }
   
    /**
        * Returns if the tree strops braching at a given node
        * @param m_Attribute the attribute of the node
        * @param level the level of the tree
        * @param data the instances of the node
        * @return if the tree stops braching
     */
    
    public boolean stopCriterion(Attribute m_Attribute, int level, Instances data){
        return m_StopLevel==-1 || Utils.grOrEq(0,infoGains[m_Attribute.index()]) || (m_StopLevel>0 && m_StopLevel==(level));
    }
    
    /* Method for building an Imprecise Credal Decision Tree.
   * @param data the training data
   * @exception Exception if decision tree can't be built successfully
   */
    
    void makeTree(NodeTree node, Instances data, int level) throws Exception {

        // Check if no instances have reached this node.
        if (data.numInstances() == 0) {
            node.setAttribute(null);
            node.setFrequency( new double[data.numClasses()]);
            return;
        }
    
        Attribute m_Attribute= this.getAttributeToRamify(data, level);
    
        // Make leaf if information gain is zero. 
        // Otherwise create successors.
        //if (this.getStopLevel()==-1 || Utils.eq(infoGains[m_Attribute.index()], 0) ||  (this.getStopLevel()>0 && this.getStopLevel()==(level))) {
        if (stopCriterion(m_Attribute, level, data)){
            node.setAttribute(null);      
            this.computeClassDistribution(node,data);      
      
        } 
        
        else {
            // Build the successors
            node.setAttribute(m_Attribute);
            Instances[] splitData = splitData(data, m_Attribute);
                   
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                NodeTree newnode=new NodeTree();
                node.setSuccesors(j,newnode);
        
                if (!m_MisclassifiedAllowed && splitData[j].numInstances()==0){
                    if (node.getFrequency()==null)
                        this.computeClassDistribution(node,data);
          
                    node.getSuccesors(j).setAttribute(null);
              //node.getSuccesors(j).setClassAttribute(data.classAttribute());

                    this.computeClassDistribution(node.getSuccesors(j),data);      

              //node.getSuccesors(j).normalize();
              //node.getSuccesors(j).setClassValue(node.getMaxDistributionIndex());
                }
                
                else
                    makeTree(node.getSuccesors(j),splitData[j],level+1);
            }
        }
    }
    
     /* Computes information gain for an attribute A in a certain node D
    * The instances are weighted depending on the weight of the corresponding class
    
    * @param data the data D for which info gain is to be computed
    * @param att the attribute A
    * @return the information gain for the given attribute and data
    */
  
    public double computeInfoGain(Instances data, Attribute att) throws Exception {
        double info_gain;
        int num_values = att.numValues();
        int num_instances = data.numInstances();
        Instances[] splitData = splitData(data, att);
        Instance instance;
        int class_value, attribute_value; 
        double entropy = computeEntropy(data);
        double conditioned_entropy;
        Instances partition;
        double weight, sum_weights_value;
        double sum_total_weights;
        double weighted_probability;
        double [] sum_weights_values = new double[num_values];
                
        
        /*
        For each possible value of the attribute a_i, consider \sum_{x \in D_i}w_j(x),
        where D_i is composed of those instances of the node for which A = a_i and 
        w_j(x) is the weight of the class value of the j-th instance of the node
        */
        
        for(int i = 0; i < num_instances; i++){
            instance = data.instance(i);
            class_value = (int) instance.classValue();
            weight = class_weights[class_value];
            attribute_value = (int) instance.value(att);
            sum_weights_values[attribute_value]+=weight;
        }
        
        sum_total_weights = Utils.sum(sum_weights_values);
        
        /*
        * Compute the Info Gain IG
        * IG = Entropy(D) - \sum_{1 = 1}^{t}P(A = ai)Entropy(D_i),
        * where t is the total number of possible values of A, 
        * Entropy(D_i) si the entropy of the partitio od D associated with A = a_I, \forall i = 1,...,t
        * and P(A = a_i) = \sum_{x \in D_i}wj(x)/\sum_{x \in D}wj(x)
        */
        
        info_gain = entropy;
        
        for (int j = 0; j < num_values; j++) {
            sum_weights_value = sum_weights_values[j];
            if (sum_weights_value > 0) {
                weighted_probability = sum_weights_value/sum_total_weights;
                partition = splitData[j];
                conditioned_entropy = computeEntropy(partition);
                info_gain -= weighted_probability * conditioned_entropy;
                
            }
        }
        
        return info_gain;
    }
    
    /**
    * Compute the entropy in a given node D
    *  Let (n_1,...,n_K) be the array with the relative frequencies in the D
    *   Let ('n_1,...,n'_K) be the arrangement associated with the probability distribution 
    *   that reaches the maximum of entropy with the A-NPI-M
    *   Let w_j the weight of the j-th class value, \foralll j = 1,...,K. 
    *   Compute p' =  (p'_1,...,p'_K), where p'i = w_ixn'_i/\sum_{j = 1}^{K}w_jxn'_j, \forall i = 1,...,K 
    *   Entropy = S(p'), being S the Shannon entropy
    *  @param data the instances of the node D 
    *  @return the entropy of the node
    *  @throws Exception 
    */
    
    protected double computeEntropy(Instances data) throws Exception {
        int num_classes = data.numClasses();
        double[] class_frequencies = new double[num_classes];
        double[] weighted_class_frequencies = new double[num_classes];
        int num_instances = data.numInstances();
        double[] NPI_transformation;
        Enumeration instEnum = data.enumerateInstances();
        Instance instance;
        int value;
        double entropy;
        double weight;
        double class_frequency, weighted_class_frequency;
        
        // Compute the class frequencies (n_1,...,n_K)
        
        while (instEnum.hasMoreElements()) {
            instance = (Instance) instEnum.nextElement();
            value = (int) instance.classValue();
            class_frequencies[value]++;
        }   
        
        /*
        *   Compute ('n_1,...,n'_K) the arrangement associated with the probability distribution 
        *   that reaches the maximum of entropy with the A-NPI-M
        */
        
        NPI_transformation = E_ContingencyTables.NPITransformation(class_frequencies);
                
        for(int j = 0; j < num_classes; j++)
            NPI_transformation[j] *= num_instances;
           
       
        // Compute p'i = w_ixn'_i/\sum_{j = 1}^{K}w_jxn'_j, \forall i = 1,...,K. 
       
        
        for(int j = 0; j < num_classes; j++){
            weight = class_weights[j];
            class_frequency = NPI_transformation[j];
            weighted_class_frequency = class_frequency*weight;
            weighted_class_frequencies[j] = weighted_class_frequency;
        }
               
        Utils.normalize(weighted_class_frequencies);
        
        
        // Compute S(p')
        
        entropy = 0;
    
        for(int j = 0; j < num_classes; j++){
            weighted_class_frequency =  weighted_class_frequencies[j];
                    
            if(weighted_class_frequency > 0)
                entropy -= weighted_class_frequency*Utils.log2(weighted_class_frequency);
        }
        
       // entropy = E_ContingencyTables.entropyNPI(weighted_class_frequencies);
        return entropy;
    }
        
    public NodeTree getRootNode(){
        return this.m_RootNode; 
    }
  
    public void setRoot(int root_index){
        m_KThRootAttribute = root_index;
    }
    
    /**
     * 
     */
    
    public boolean[] computeNonDominatedStatesSet(Instance instance) throws NoSupportForMissingValuesException{
        double[] frequency=this.frequencyForInstance(instance);
        boolean[] non_dominated_states_set = computeNonDominatedSet(frequency);
        
        return non_dominated_states_set;
    }
    
    /**
    * This function updates the evaluation metrics of the Imprecise Classifier given an instance 
    * @param instance the given instance
    * @throws NoSupportForMissingValuesException 
    */
  
    private void updateCredalStatistics(Instance instance) throws NoSupportForMissingValuesException{

        boolean[] non_dominated_states_set  = this.computeNonDominatedStatesSet(instance);       

        int[] nonDominatedIndexSet = getNonDominatedIndexSet(non_dominated_states_set);


        this.updateStatistics(nonDominatedIndexSet, instance);

    }
    
     /**
    * Compute the non dominates states set given the relative frequencies of the class values
    * It computes the weighted lower and upper probabilites 
    * Let (n_1,...,n_K) be the relative frequencies of the class values
    * According to the A-NPI-M, \\underline{p_i} = max(0, (n_i-1)/N),  \\overline{p_i} = min(1, (n_i+1)/N), \forall i = 1,...,K
    * Consider the weighted probabilities w\\underline{p_i} = wi\\underline{p_i} and w\\overline{p_i} = wi\\overline{p_i} \forall i = 1,...,K
    * being wi the weight associated with the i-th class value
    * it uses the stochastic dominance crtierion based on the weighted lower and upper probabilites 
    * cj dominates ci iff w\\underline{p_j} >= w\\overline{p_i}   
    * @param frequencies the relative frequencies of the class values
    * @return a boolean vector, where the i-th component is equal to true if the i-th class value is non-dominated
    */

    public boolean[] computeNonDominatedSet(double[] frequencies) {
        int num_class_values = frequencies.length;
        double num_instances = Utils.sum(frequencies);
        double[] weighted_frequencies;
        double sum_weights;
        boolean[] non_dominated_set;
        double[] lower_probabilities = new double[num_class_values];
        double[] upper_probabilities = new double[num_class_values];
        double frequency, weighted_frequency;
        double lower_probability, upper_probability;
        double weight;
        
        // Compute w_i\times n_i \forall i = 1,...,K
        
        weighted_frequencies = computeWeightedFrequencies(frequencies);
        
        // Consider \sum_{i = 1}^{K} w_i\times n_i
        
        sum_weights = Utils.sum(weighted_frequencies);
        
        /*
        Compute w\\underline{p_i} = max(0, wi x (n_i-1)/(\sum_{j = 1}^{N}n_jxw_j)), \forall i = 1,...,N
                w\\overline{p_i} = min(1, wi x (n_i+1)/(\sum_{j = 1}^{N}n_jxw_j))), \forall i = 1,...,N
        */
        
        for(int k = 0; k < num_class_values; k++){
            weight = class_weights[k];
            frequency = frequencies[k];
            
            if(frequency <= 1)
                lower_probability = 0;
            
            else{
                weighted_frequency = weight*(frequency-1);
                lower_probability = weighted_frequency/sum_weights;
            }
  
            if(frequency < num_instances){
                weighted_frequency = weight*(frequency+1);
                upper_probability = weighted_frequency/sum_weights;   
            }
            
            else
                upper_probability = 1;
            
           
            lower_probabilities[k] = lower_probability;
            upper_probabilities[k] = upper_probability;
        }
        
        /*
       Obtain the non-dominated states set from these probabilities
        */
        
        non_dominated_set = computeNonDominatedSetFromProbabilities(lower_probabilities, upper_probabilities);
  
        return non_dominated_set;
    }
    
    /* Main method.
    *
    * @param args the options for the classifier
    */
  
    public static void main(String[] args) {

        try {
            System.out.println(Evaluation.evaluateModel(new CredalDecisionTree_CostErrors_Weighted(), args));
        } catch (Exception e) {
        System.err.println(e.getMessage());
        }
    }
 
}
