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


public class CredalDecisionTreeDirectNPI  extends CredalClassifier implements OptionHandler, AdditionalMeasureProducer{
    
  /* Field that contains the tree structure*/
  NodeTree m_RootNode;
  
   /** Constant values that indicate the split criterion that it is used to build the decision tree.*/
  public static final int CORRECT_INDICATION=0;
   
  /** This field stores the index of the correct indication criterium used to build the decision tree*/
  int m_SplitMetric=CORRECT_INDICATION;
  
  static final String[] STRING_SPLIT_METRIC={"DIRECT_NPI"};
  public static final Tag[] TAGS_SPLIT_METRIC =
    {
        new Tag(CORRECT_INDICATION, STRING_SPLIT_METRIC[0]),
   };
  
   /* if this field is set to false, the class of a leave where the number of occurrences of the configuration
        that defines is zero is set to the most probable class that defines its parent if if was a leave */
  boolean m_MisclassifiedAllowed=true;
  
  /** The highest NPI lower probability for a class value without conditioning to any attribute */
  
  double inferior_probability_no_test;
  
    /** The highest NPI upper probability for a class value without conditioning to any attribute */
  
  double superior_probability_no_test;
  
  /** The lower probabilities of correct indications for all the attributes */
  
  double[] lower_probabilities_correct_indication;
  
  /** The upper probabilities of correct indications for all the attributes */
  
  double[] upper_probabilities_correct_indication;
  
  /* Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation   result;

    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Abellan, J. and Moral, S");
    result.setValue(Field.YEAR, "2003");
    result.setValue(Field.TITLE, "Building classiffication trees using the total uncertainty criterion");
    result.setValue(Field.JOURNAL, "International Journal of Intelligent Systems");
    result.setValue(Field.VOLUME, "18");
    result.setValue(Field.NUMBER, "12");
    result.setValue(Field.PAGES, "1215-1225");

    return result;
  }
  
  /* Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

// instances
    result.setMinimumNumberInstances(0);
    
    return result;
  }
  
   /**
    * Returns a superconcise version of the model
    * @return the summary of the model
    */
  
    public String toSummaryString() {

        return "Number of leaves: " + numLeaves(m_RootNode) + "\n"
         + "Size of the tree: " + numNodes(m_RootNode) + "\n";
    }
  
    /**
    * It obtains the number of leaves in a tree node
    * @param node the node
    * @return  the number of leaves in the node. 
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
    
    /**
     * It obtains the number of nodes in a node tree structure 
     * @param node the node structure
     * @return the number of nodels 
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

    /**
    * Returns the number of leaves
    * @return the number of leaves
    */
  
    public double measureNumLeaves() {
        return numLeaves(this.m_RootNode);
    }
    
    /**
     * Returns an enumeration of the additional measure names
    * @return an enumeration of the measure names
    */
    
    public Enumeration enumerateMeasures() {
        Vector newVector = new Vector(3);
        newVector.addElement("measureTreeSize");
        newVector.addElement("measureNumLeaves");

    
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
       
        if (additionalMeasureName.compareToIgnoreCase("measureTreeSize") == 0) 
            return measureTreeSize();
        
        else if (additionalMeasureName.compareToIgnoreCase("measureNumLeaves") == 0) 
            return measureNumLeaves();

        return super.getMeasure(additionalMeasureName);

    }
    
    /**
    * Returns a string describing the classifier.
    * @return a description suitable for the GUI.
    */
    
    public String globalInfo() {

        String output = "Method for building a decision tree for Imprecise Classification based on direct NPI classification."
        +"It uses the Correct Indication measure (CI) as the split criterion, secting the attribute with the highest lower and upper probability of correct indication."
        +"The rest of the model (the criterion used for classify the instances in the leaf nodes) is equal to the alreasy existing ICDT."
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
     * Checks whether all the instances have the same value for a certain attribute
     * (Whether the entropy of that attribute is equal to 0)
     * @param data the instances 
     * @param attribute_index the index of the corresponding value
     * @return a boolean value, true iif all the instances of data have the same value of the attribute attribute_index
     */
    
    private boolean zeroEntropy(Instances data, int attribute_index){
        boolean zero_entropy = true;
        Instance instance = data.instance(0);
        int previous_value = (int)instance.value(attribute_index);
        int current_value;
        int i = 1;
        
        while(zero_entropy){
            instance = data.instance(i);
            current_value = (int)instance.value(attribute_index);
            zero_entropy = current_value == previous_value;
            
            if(zero_entropy){
                previous_value = current_value;
                i++;
            }
        }
        
        return zero_entropy;
    }
    
  /**
   * It determines the attribute for spliting in a certain node of the tree
   * @param data the instances in the node
   * @return the selected attribute for splitting
   */
    
    protected Attribute getAttributeToRamify(Instances data){
        int num_instances = data.numInstances();
        int num_class_values = data.numClasses();
        int num_attributes = data.numAttributes();
        int num_values;
        Instance instance; 
        int class_value, attribute_value;
        // Compute the frequencies of the class values in the given set of instances
        double[] class_frequencies = new double[num_class_values];
        double[][] class_attribute_frequencies;
        double superior_probability, inferior_probability;
        double class_frequency, attribute_frequency, class_attribute_frequency;
        int index_selected_attribute;
        Attribute selected_attribute;
        boolean splitted;
        double[] bernouilli_NPI;
        double bernouilli_probability;
        int[] sort_indices_bernouilli_NPI;
        int sort_index;
        double probability_inferior, probability_superior;
        int next_index, max_index;
        
        // Compute the frequencies of the class values
        
        for(int i = 0; i < num_instances; i++){
            instance = data.instance(i);
            class_value = (int)instance.classValue();
            class_frequencies[class_value]++;
        }
        
        /*
        Compute the lower and upper inferior prior probabilities
        The maximum and minimum NPI probabilities of the class values without conditioning
        */
        inferior_probability_no_test = Double.NEGATIVE_INFINITY;
        superior_probability_no_test = Double.NEGATIVE_INFINITY;
        
        for(int i = 0; i < num_class_values; i++){
            class_frequency = class_frequencies[i];
            
        // lower NPI-M probability
            if(class_frequency > 0)
                inferior_probability = (class_frequency-1)/num_instances;
            
            else
                inferior_probability = 0;
            
            // Upper NPI-M probability
            if(class_frequency < num_instances)
                superior_probability = (class_frequency + 1)/num_instances;
            
            else
                superior_probability = 1;
            
            // Update the maximum and minimum lower probabilities
            if(inferior_probability > inferior_probability_no_test)
                inferior_probability_no_test = inferior_probability;
            
            if(superior_probability > superior_probability_no_test)
                superior_probability_no_test = superior_probability;
        }
        
        lower_probabilities_correct_indication = new double[num_attributes];
        upper_probabilities_correct_indication = new double[num_attributes];
        
        for(int i = 0; i < num_attributes; i++){
            // Check wheter the attribute has been selected for splitting 
            splitted = zeroEntropy(data, i);
            
            /* If it has been selected for splitting, then the lower and upper 
            probabilities of correct indication are equal to -infinity, in this case
            the attribute is not candidate
            */
            if(splitted){
                lower_probabilities_correct_indication[i] = Double.NEGATIVE_INFINITY;
                upper_probabilities_correct_indication[i] = Double.NEGATIVE_INFINITY;
            }
            
            else{
                num_values = data.attribute(i).numValues();
                class_attribute_frequencies = new double[num_values][num_class_values];
                
                /*
                Compute the contingency matrix
                1 row for each possible value of the attribute and one colum for each class value
                */
                for(int j = 0; j < num_instances; j++){
                    instance = data.instance(i);
                    class_value = (int)instance.classValue();
                    attribute_value = (int) instance.value(i);
                    class_attribute_frequencies[attribute_value][class_value]++;
                }
                
                /*
                Compute, for each possible value of the attribute c_i,
                nCi(tj = ci)/n(tj = ci)+1
                */
                
                bernouilli_NPI = new double[num_values];
                
                for(int j = 0; j < num_values; j++){
                    attribute_frequency = Utils.sum(class_attribute_frequencies[j]);
                    max_index = Utils.maxIndex(class_attribute_frequencies[j]);
                    class_attribute_frequency = class_attribute_frequencies[j][max_index];
                    bernouilli_probability = class_attribute_frequency/(attribute_frequency + 1);
                    bernouilli_NPI[j] = bernouilli_probability;
                }
                
                /*  Sort this array in order to compute the pi that maximizes for the upper probablity of CI
                    as well as the prob the minimizes the lower probability for CI,
                    taking into account that pi \in [ni -1/n, ni+1/n]
                */
                
                sort_indices_bernouilli_NPI = Utils.sort(bernouilli_NPI);
                /**
                 * The first k/2 with highest Bernouilli probability have assign 
                 * the min NPI lower probability for lower probability of correct indication 
                 * and the max NPI upper probability for upper probability of correct indication
                 */
                for(int j = 0; j < num_values/2; j++){
                    sort_index = sort_indices_bernouilli_NPI[j];
                    attribute_frequency = Utils.sum(class_attribute_frequencies[sort_index]);
                    probability_inferior = (attribute_frequency - 1)/num_instances;
                    
                    if(probability_inferior < 0)
                        probability_inferior = 0;
                    
                     probability_superior = (attribute_frequency + 1)/num_instances;
                    
                    if(probability_superior > 1)
                        probability_superior = 1;
                    
                    lower_probabilities_correct_indication[i] += probability_inferior*bernouilli_NPI[sort_index];
                    upper_probabilities_correct_indication[i] += probability_superior*bernouilli_NPI[sort_index];
                }
                
                /*
                If the num of values is odd, then the value with the middle bernouilli probability
                has a probabability estimated by relative frequencies 
                for both lower and upper probability of correct indication
                */
                
                if(num_values % 2 == 1){
                    sort_index = sort_indices_bernouilli_NPI[num_values/2];
                    attribute_frequency = Utils.sum(class_attribute_frequencies[sort_index]);
                    probability_inferior = attribute_frequency/num_instances;
                    lower_probabilities_correct_indication[i] += probability_inferior*bernouilli_NPI[sort_index];
                    upper_probabilities_correct_indication[i] += probability_inferior*bernouilli_NPI[sort_index];
                    next_index = num_values / 2 + 1;
                }
                
                else
                    next_index = num_values/2;
                
                 /**
                 * The last k/2 with highest Bernouilli probability have assign 
                 * the max NPI lower probability for lower probability of correct indication 
                 * and the min NPI upper probability for upper probability of correct indication
                 */
                
                for(int j = next_index; j < num_values/2; j++){
                    sort_index = sort_indices_bernouilli_NPI[j];
                    attribute_frequency = Utils.sum(class_attribute_frequencies[sort_index]);
                    probability_inferior = (attribute_frequency + 1)/num_instances;
                    
                    if(probability_inferior > 0)
                        probability_inferior = 1;
                    
                     probability_superior = (attribute_frequency - 1)/num_instances;
                    
                    if(probability_superior < 0)
                        probability_superior = 0;
                    
                    lower_probabilities_correct_indication[i] += probability_inferior*bernouilli_NPI[sort_index];
                    upper_probabilities_correct_indication[i] += probability_superior*bernouilli_NPI[sort_index];
                }
                    
            }
                
        }
        
        /* Now, we select the attribute for splitting
         * 
         */
        
        index_selected_attribute = -1;
        
        if(index_selected_attribute != -1)
            selected_attribute = data.attribute(index_selected_attribute);
        
        else
            selected_attribute = null;
            
        return selected_attribute;
    }
    
   /* Builds a decision tree imprecise classifier based on direct NPI classification.
   *
   * @param data the training data
   * @exception Exception if classifier can't be built successfully
   */
    
    public void buildClassifier(Instances data) throws Exception {
        initStats();
    
        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
    
        this.m_RootNode=new NodeTree();
    
        makeTree(this.m_RootNode,data);
    }
    
    /* Computes class distribution for instance using decision tree.
    *
    * @param instance the instance for which distribution is to be computed
    * @return the class distribution for the given instance
    * @throws NoSupportForMissingValuesException if instance has missing values
    */
  
    public double[] distributionForInstance(Instance instance) throws Exception {
        int num_class_values = instance.numClasses();
        
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("ICDT based on direct NPI classification: no missing values, "
                                                   + "please.");
        }

        this.updateCredalStatistics(instance);

        double [] probs = new double[num_class_values];
    
        for (int i=0; i<probs.length; i++)
            probs[i]=Double.NaN;

        return probs;
    }

    /** Computes class distribution for instance using decision tree.
    *
    * @param instance the instance for which distribution is to be computed
    * @return the class distribution for the given instance
    * @throws NoSupportForMissingValuesException if instance has missing values
    */
  public double[] frequencyForInstance(Instance instance) 
    throws NoSupportForMissingValuesException {

    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("ICDT based on direct NPI classification: no missing values, " + "please.");
    }
    
    return this.frequencyForInstance(this.m_RootNode, instance);

  }
  
    /**
    * Computes the class distribution for the instance in a certain node
    * @param node the node in which the distribution is computed
    * @param instance the instance for which distribution is to be computed
    * @return the class distribution for the given instance in the given node
    */
  
    private double[] frequencyForInstance(NodeTree node, Instance instance){
      
        if (node.getAttribute()== null) {
            return node.getFrequency();
        } 
    
        else { 
            return frequencyForInstance(node.getSuccesors((int) instance.value(node.getAttribute())), instance);
        }
      
    }
    
    /** Computes class distribution for instance using decision tree.
    *
    * @param instance the instance for which distribution is to be computed
    * @return the class distribution for the given instance
    * @throws NoSupportForMissingValuesException if instance has missing values
    */
  
    public double ocurrencesForInstance(Instance instance) throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) 
            throw new NoSupportForMissingValuesException("ICDT based on direct NPI classification: no missing values, " + "please.");
    
        return ocurrencesForInstance(this.m_RootNode,instance);
    }
    
    /** Computes class distribution for instance using decision tree in a certain node.
    * @param node the node in which the distribution is computed
    * @param instance the instance for which distribution is to be computed
    * @return the class distribution for the given instance
    */
    
    private double ocurrencesForInstance(NodeTree node, Instance instance){
        
        if (node.getAttribute()== null) 
            return 1;
     
        else  
            return node.getAttribute().numValues()*ocurrencesForInstance(node.getSuccesors((int) instance.value(node.getAttribute())),instance);
      
    }
  
    /** Classifies a given test instance using the decision tree.
    * @param instance the instance to be classified
   * @return the classification
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("ICDT based on direct NPI classification: no missing values, " + "please.");
        }

        this.updateCredalStatistics(instance);

        return Double.NaN;
    }
    
    /**
     * Splits a dataset according to the values of a nominal attribute.
    * @param data the data which is to be split
    * @param att the attribute to be used for splitting
    * @return the sets of instances produced by the split
    */
    protected Instances[] splitData(Instances data, Attribute att) {
        int num_values = att.numValues();
        int num_instances = data.numInstances();
        Instances[] splitData = new Instances[num_values];
        int attribute_value;
    
        for (int j = 0; j < att.numValues(); j++) 
            splitData[j] = new Instances(data, num_instances);
        
        Enumeration instEnum = data.enumerateInstances();
        
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            attribute_value = (int) inst.value(att);
            splitData[attribute_value].add(inst);
        }
        
        for (int i = 0; i < splitData.length; i++) 
            splitData[i].compactify();
    
        return splitData;
    }
    
    /** Computes frequencies for instance using decision tree in a certain node.
    * @param node the node in which the frequencies are computed
    * @param instance the instance for which frequencies are to be computed
    * @return the frequencies for the given instance
    */
    
    
    protected void computeFrequencies(NodeTree node, Instances data){
        int class_value;
        
        node.setFrequency(new double[data.numClasses()]);
        Enumeration instEnum = data.enumerateInstances();
      
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            class_value = (int) inst.classValue();
            node.getFrequency()[class_value]++;
      }
      //node.setSupport(data.numInstances());

    }
    
    /**
    * Method for building the decision tree based o direct NPI classification in a certain node 
    * @param node the node at which the tree is built
    * @param data the instances in the node
    * @throws Exception if decision tree can't be built successfully
    */
    void makeTree(NodeTree node, Instances data) throws Exception {
        int num_instances = data.numInstances();
        int num_classes = data.numClasses();
        // Check if no instances have reached this node.
        
        if (num_instances == 0) {
            node.setAttribute(null);
            node.setFrequency(new double[num_classes]);
            return;
        }   

        //node.setClassAttribute(data.classAttribute());
    
        Attribute m_Attribute= this.getAttributeToRamify(data);
    
        /* Make leaf if there is no an attribute for which the lower and upper probabilities of the correct indications are greater 
           than the highest lower and upper probabilities of the class values without conditioning
            */ 
        // Otherwise create successors.

        if (m_Attribute == null){
      
            node.setAttribute(null);
      //node.setClassAttribute(data.classAttribute());
            this.computeFrequencies(node,data);      
      
        } 
        
        else {
            node.setAttribute(m_Attribute);
            
            Instances[] splitData = splitData(data, m_Attribute);
          
          
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                NodeTree newnode=new NodeTree();
                node.setSuccesors(j,newnode);
        
                if (!m_MisclassifiedAllowed && splitData[j].numInstances()==0){
                    if (node.getFrequency()==null)
                        this.computeFrequencies(node,data);
          
                    node.getSuccesors(j).setAttribute(null);

                    this.computeFrequencies(node.getSuccesors(j),data);      
                }
                
                else
                    makeTree(node.getSuccesors(j),splitData[j]);
            }
        }
    }
    
   /**
    * @return the root node
    * @return 
    */
    
    public NodeTree getRootNode(){
        return this.m_RootNode; 
    }
  
    /** 
    * This function updates the credal statistics given an instance 
    * @param instance the instance
    */
    private void updateCredalStatistics(Instance instance) throws NoSupportForMissingValuesException{

        double[] frequency=this.frequencyForInstance(instance);

        boolean[] nonDominatedSet= computeNonDominatedSet(frequency);

        int cont=0;
      
        for (int i=0; i<nonDominatedSet.length; i++){
            if (nonDominatedSet[i])
                cont++;
        }

        int[] nonDominatedIndexSet = new int[cont];

        cont=0;
     
        for (int i=0; i<nonDominatedSet.length; i++){
            if (nonDominatedSet[i])
                nonDominatedIndexSet[cont++]=i;
        }

        this.updateStatistics(nonDominatedIndexSet, instance);

    }

    /**
     * It computes the 
     * @param frequency the frequencies of the insnce
     * @return a boolean vector, where the i-th component indicates whether the i-th class value is non dominated.  
     */
    public boolean[] computeNonDominatedSet(double[] frequency) {

        int num_class_values = frequency.length;
        double num_instances=Utils.sum(frequency);
        boolean[] non_dominates_set= new boolean[num_class_values];
        double max_inferior_probability;
        double inferior_probability, superior_probability;
        double denominator = num_instances + this.m_SValue;
        
        for (int k=0; k<frequency.length; k++){
            max_inferior_probability=Double.NEGATIVE_INFINITY;

             for (int i=0; i<frequency.length; i++){
                if (i!=k){
                    inferior_probability=frequency[i]/denominator;

                if (inferior_probability>max_inferior_probability)
                    max_inferior_probability=inferior_probability;
            }
        }

        superior_probability=(frequency[k]+this.m_SValue)/denominator;

        non_dominates_set[k] = superior_probability>max_inferior_probability;

    }
    return non_dominates_set;
  }


   /* Main method.
   *
   * @param args the options for the classifier
   */
    public static void main(String[] args) {

        try {
        System.out.println(Evaluation.evaluateModel(new CredalDecisionTreeDirectNPI(), args));
        }   catch (Exception e) {
        System.err.println(e.getMessage());
        }
    }
  
}
