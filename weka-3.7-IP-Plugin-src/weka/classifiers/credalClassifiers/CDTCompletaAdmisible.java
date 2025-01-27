/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.credalClassifiers;

import java.util.Enumeration;
import java.util.Vector;
import weka.classifiers.Evaluation;
import static weka.classifiers.credalClassifiers.CDTCompletaAdmisible.IMPRECISE_ENTROPY;
import weka.classifiers.trees.NodeTree;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.E_ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.Utils;

/**
 *
 * @author Serafin
 */
public class CDTCompletaAdmisible extends CredalClassifier implements OptionHandler, AdditionalMeasureProducer{
    /* Field that contains the tree structure*/
  NodeTree m_RootNode;
  
  /* Level of the tree when the building of tree stops altought there is an improvement in the entropy. 
     If it sets to 0, the unique stop criterium is the criterium of deterioration of the entropy*/
  int m_StopLevel=0; 
  
  /* Constant values that indicate the entropy that it is used to build the decision tree.*/
  public static final int IMPRECISE_ENTROPY=0;
   
  /* This field stores the index of the entropy criterium used to build the decision tree */
  int m_SplitMetric=IMPRECISE_ENTROPY;
  
  
  static final String[] STRING_SPLIT_METRIC={"IDM"};
  public static final Tag[] TAGS_SPLIT_METRIC =
    {
        new Tag(IMPRECISE_ENTROPY, STRING_SPLIT_METRIC[0]),
   };
  
  
  /* if this field is set to false, the class of a leave where the number of occurrences of the configuration
        that defines is zero is set to the most probable class that defines its parent if if was a leave */
  boolean m_MisclassifiedAllowed=true;
  
  /* Vector with the ingoGain value of each attribute*/
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

    result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
    result.setValue(TechnicalInformation.Field.AUTHOR, "Abellan, J. and Moral, S");
    result.setValue(TechnicalInformation.Field.YEAR, "2003");
    result.setValue(TechnicalInformation.Field.TITLE, "Building classiffication trees using the total uncertainty criterion");
    result.setValue(TechnicalInformation.Field.JOURNAL, "International Journal of Intelligent Systems");
    result.setValue(TechnicalInformation.Field.VOLUME, "18");
    result.setValue(TechnicalInformation.Field.NUMBER, "12");
    result.setValue(TechnicalInformation.Field.PAGES, "1215-1225");

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
    result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

    // class
    result.enable(Capabilities.Capability.NOMINAL_CLASS);
    result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

// instances
    result.setMinimumNumberInstances(0);
    
    return result;
  }
  
  
  /* Returns a superconcise version of the model
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
  public double measureNumRules() {
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
    } else if (additionalMeasureName.compareToIgnoreCase("measureTreeSize") == 0) {
      return measureTreeSize();
    } else if (additionalMeasureName.compareToIgnoreCase("measureNumLeaves") == 0) {
      return measureNumLeaves();
    }

    return super.getMeasure(additionalMeasureName);

}
 
  /* Returns a string describing the classifier.
   * @return a description suitable for the GUI.
   */
  public String globalInfo() {

    String output = " Method for building an unpruned decision tree based on the ID3"
            +"algorithm. It applies a the measure of entropy for credal sets as a branching criterion for constructing the classification trees."
            +"This branching criterion is based on imprecise probabilities which are determined with the imprecise Dirichlet model."
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

  
  protected Attribute getAttributeToRamify(Instances data, int level) throws Exception{
    
    // Compute attribute with maximum information gain.
    infoGains = new double[data.numAttributes()];
    Enumeration attEnum = data.enumerateAttributes();
    while (attEnum.hasMoreElements()) {
      Attribute att = (Attribute) attEnum.nextElement();
      infoGains[att.index()] = computeInfoGain(data, att);
    }
    infoGains[data.classIndex()]=-Double.MAX_VALUE;
    
    if (level==0){
        return data.attribute(Utils.sort(infoGains)[infoGains.length-this.m_KThRootAttribute]);
    }else{
        return data.attribute(Utils.maxIndex(infoGains));
    }

  }

  /* Builds IPTree decision tree classifier.
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
    
    makeTree(this.m_RootNode,data,0);
  }

  /* Computes class distribution for instance using decision tree.
   *
   * @param instance the instance for which distribution is to be computed
   * @return the class distribution for the given instance
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
  public double[] distributionForInstance(Instance instance) 
    throws Exception {

    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("IPTree: no missing values, "
                                                   + "please.");
    }

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
  public double[] frequencyForInstance(Instance instance) 
    throws NoSupportForMissingValuesException {

    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("IPTree: no missing values, "
                                                   + "please.");
    }
    
    return this.frequencyForInstance(this.m_RootNode, instance);

  }

private double[] frequencyForInstance(NodeTree node, Instance instance){
      
    if (node.getAttribute()== null) {
      return node.getFrequency();
    } else { 
      return frequencyForInstance(node.getSuccesors((int) instance.value(node.getAttribute())), instance);
    }
      
  }
  
  
  /* Computes class distribution for instance using decision tree.
   *
   * @param instance the instance for which distribution is to be computed
   * @return the class distribution for the given instance
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
  public double ocurrencesForInstance(Instance instance) 
    throws NoSupportForMissingValuesException {

    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("IPTree: no missing values, "
                                                   + "please.");
    }
    return ocurrencesForInstance(this.m_RootNode,instance);
  }  
  
  private double ocurrencesForInstance(NodeTree node, Instance instance){
    if (node.getAttribute()== null) {
      return 1;
    } else { 
      return node.getAttribute().numValues()*ocurrencesForInstance(node.getSuccesors((int) instance.value(node.getAttribute())),instance);
    }
      
  }
  
  /* Classifies a given test instance using the decision tree.
   *
   * @param instance the instance to be classified
   * @return the classification
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
  public double classifyInstance(Instance instance) 
    throws NoSupportForMissingValuesException {

    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("IPTree: no missing values, "
                                                   + "please.");
    }

    this.updateCredalStatistics(instance);

    return Double.NaN;
  }
  

  
  /* Splits a dataset according to the values of a nominal attribute.
   *
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
   
  public boolean stopCriterion(Attribute m_Attribute, int level, Instances data){
      return m_StopLevel==-1 || Utils.grOrEq(0,infoGains[m_Attribute.index()]) || (m_StopLevel>0 && m_StopLevel==(level));
  }
  
  /* Method for building an IPTree tree.
   *
   * @param data the training data
   * @exception Exception if decision tree can't be built successfully
   */
  void makeTree(NodeTree node, Instances data, int level) throws Exception {

    // Check if no instances have reached this node.
    if (data.numInstances() == 0) {
      node.setAttribute(null);
      //node.setClassAttribute(data.classAttribute());      
      //node.setClassValue(Instance.missingValue());
      //node.setDistribution( new double[data.numClasses()]);
      node.setFrequency( new double[data.numClasses()]);
      //node.setSupport(0);
      return;
    }

//node.setClassAttribute(data.classAttribute());
    
    Attribute m_Attribute= this.getAttributeToRamify(data, level);
    
    // Make leaf if information gain is zero. 
    // Otherwise create successors.
    //if (this.getStopLevel()==-1 || Utils.eq(infoGains[m_Attribute.index()], 0) ||  (this.getStopLevel()>0 && this.getStopLevel()==(level))) {
    if (stopCriterion(m_Attribute, level, data)){
      
      node.setAttribute(null);
      //node.setClassAttribute(data.classAttribute());
      
      this.computeClassDistribution(node,data);      
      
      //node.normalize();
      //node.setClassValue(node.getMaxDistributionIndex());
    } else {

      node.setAttribute(m_Attribute);
     
      //this.computeClassDistribution(node,data);        
      
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
        }else{
            makeTree(node.getSuccesors(j),splitData[j],level+1);
        }
        
      }
    }
  }


  /* Computes information gain for an attribute.
   *
   * @param data the data for which info gain is to be computed
   * @param att the attribute
   * @return the information gain for the given attribute and data
   */
  public double computeInfoGain(Instances data, Attribute att) 
    throws Exception {

    double infoGain = computeEntropy(data);
    Instances[] splitData = splitData(data, att);
    for (int j = 0; j < att.numValues(); j++) {
          if (splitData[j].numInstances() > 0) {
            infoGain -= ((double) splitData[j].numInstances() /
                         (double) data.numInstances()) *
              computeEntropy(splitData[j]);
          }
    }
    return infoGain;
  }

  /* Computes the entropy of a dataset.
   * 
   * @param data the data for which entropy is to be computed
   * @return the entropy of the data's class distribution
   */
  protected double computeEntropy(Instances data) throws Exception {

    double [] classCounts = new double[data.numClasses()];
    Enumeration instEnum = data.enumerateInstances();
    while (instEnum.hasMoreElements()) {
      Instance inst = (Instance) instEnum.nextElement();
      classCounts[(int) inst.classValue()]++;
    }
    
    if (this.m_SplitMetric==CDTCompletaAdmisible.IMPRECISE_ENTROPY)
        return E_ContingencyTables.entropyImprecise(classCounts,this.getSValue());
    else
        return 0.0;
        
  }
    
  
  public NodeTree getRootNode(){
      return this.m_RootNode; 
  }
  
  public void setRoot(int root_index){
      m_KThRootAttribute = root_index;
  }

  /* 
   * This function computes a utility measure of an IP model
   * 
   */
  private void updateCredalStatistics(Instance instance) throws NoSupportForMissingValuesException{
      boolean[] nonDominatedSet= computeNonDominatedStatesSet(instance);

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
   * Compute the non-dominated states set for an instance according to complete-admissible dominance criterion
   * @param instance Instance
   * @return array K-dimensional. Component i = true iif state i non-dominated
   */
  
    public boolean[] computeNonDominatedStatesSet(Instance instance) throws NoSupportForMissingValuesException{
        boolean[] non_dominated_states;
        double[] frequencies=this.frequencyForInstance(instance);
        double frequency;
        double max_frequency = Double.NEGATIVE_INFINITY;
        int num_classes = instance.numClasses();
        boolean non_dominated;
        
        non_dominated_states = new boolean[num_classes];
        
        for(int i = 0; i < num_classes; i++){
            frequency = frequencies[i];
            
            if(frequency > max_frequency)
                max_frequency = frequency;
        }
        
        for(int i = 0; i < num_classes; i++){
            frequency = frequencies[i];
            non_dominated = frequency == max_frequency;
            non_dominated_states[i] = non_dominated;
        }
   
        return non_dominated_states;
    }
    

  /* Main method.
   *
   * @param args the options for the classifier
   */
  public static void main(String[] args) {

    try {
      System.out.println(Evaluation.evaluateModel(new CDTCompletaAdmisible(), args));
    } catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }

    
}
