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

package weka.classifiers.trees;

import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Capabilities.Capability;
import weka.classifiers.*;
import weka.core.*;
import java.util.*;

/**
 * Class for constructing an unpruned decision tree based on the ID3 algorithm. The Information Gain metric is used as well as
 * another metrics like GiniIndex, Information Gain Ratio and a new entropy metric based on imprecise probabilitities.
 */
public class IPTree extends Classifier implements OptionHandler, AdditionalMeasureProducer{

  /** for serialization */
  static final long serialVersionUID = 2889730616939923301L;

  /** Parameter for the IDM**/
  protected double m_SValue = 1.0;


  /** Field that contains the tree structure*/
  NodeTree m_RootNode;
  
  /** Level of the tree when the building of tree stops altought there is an improvement in the entropy. 
   *  If it sets to 0, the unique stop criterium is the criterium of deterioration of the entropy*/
  int m_StopLevel=0; 
  
  /** Constant values that indicate the entropy that it is used to build the decision tree.*/
  public static final int ENTROPY=0;
  public static final int IMPRECISE_ENTROPY=1;
  public static final int GINI_INDEX=2;
  public static final int J48_INFOGAINRATIO=3;

  /** This field stores the index of the entropy criterium used to build the decision tree*/
  int m_SplitMetric=IMPRECISE_ENTROPY;
  
  
  static final String[] STRING_SPLIT_METRIC={"Entropy","IDM","GiniIndex","J48InfoGainRatio"};
  public static final Tag[] TAGS_SPLIT_METRIC =
    {
        new Tag(ENTROPY, STRING_SPLIT_METRIC[0]),
        new Tag(IMPRECISE_ENTROPY, STRING_SPLIT_METRIC[1]),
        new Tag(GINI_INDEX, STRING_SPLIT_METRIC[2]),
        new Tag(J48_INFOGAINRATIO, STRING_SPLIT_METRIC[3]),
  };
  
  
  /** if this field is set to false, the class of a leave where the number of occurrences of the configuration
        that defines is zero is set to the most probable class that defines its parent if if was a leave*/
  boolean m_MisclassifiedAllowed=false;
  
  /** Vector with the ingoGain value of each attribute*/
  double[] infoGains;
  
  static final String[] STRING_LEAF_ESTIMATION={"MLE","LAPLACE"};
  public static final Tag[] TAGS_LEAF_ESTIMATION =
    {
        new Tag(0, STRING_LEAF_ESTIMATION[0]),
        new Tag(1, STRING_LEAF_ESTIMATION[1]),
    };

  int m_LeafEstimation=1;


  /** If this field contains a value k then the attribute with k-th hightest value of Information Gain will be used
      as a root of the node. The default value is k=1.*/
  int m_KThRootAttribute=1;

 /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;

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
  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
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
  
  
  /**
   * Returns a superconcise version of the model
   */
  
  public String toSummaryString() {

    return "Number of leaves: " + numLeaves(m_RootNode) + "\n"
         + "Size of the tree: " + numNodes(m_RootNode) + "\n";
  }
   
  /**
   * Returns number of leaves in tree structure.
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
   * Returns number of nodes in tree structure.
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

  /**
   * Returns the size of the tree
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
   * Returns the number of rules (same as number of leaves)
   * @return the number of rules
   */
  public double measureNumRules() {
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
    newVector.addElement("measureNumRules");

    return newVector.elements();
  }

  /**
   * Returns the value of the named measure
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

    throw new IllegalArgumentException(additionalMeasureName+ " not supported IPTree");

  }
 
  /**
   * Returns a string describing the classifier.
   * @return a description suitable for the GUI.
   */
  public String globalInfo() {

    return  " Class for constructing an unpruned decision tree based on the ID3 algorithm."
            + "Several split scores can be employed: the Information Gain, GiniIndex,"
            + "Information Gain Ratio and scores based on imprecise probabilitities."
            + "For more information, see:\n\n"
            + getTechnicalInformation().toString();
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
        return data.attribute(Utils.sort(infoGains)[infoGains.length-this.getKTHRootAttribute()]);
    }else{
        return data.attribute(Utils.maxIndex(infoGains));
    }

  }

  /**
   * Builds IPTree decision tree classifier.
   *
   * @param data the training data
   * @exception Exception if classifier can't be built successfully
   */
  public void buildClassifier(Instances data) throws Exception {

   
    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    this.m_RootNode=new NodeTree();
    
    makeTree(this.m_RootNode,data,0);
  }

  /**
   * Computes class distribution for instance using decision tree.
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
    return this.distributionForInstance(this.m_RootNode,instance);
  }
  
  private double[] distributionForInstance(NodeTree node, Instance instance){
      
    if (node.getAttribute()== null) {
      return node.getDistribution();
    } else { 
      return distributionForInstance(node.getSuccesors((int) instance.value(node.getAttribute())), instance);
    }
      
  }
  
  /**
   * Computes class distribution for instance using decision tree.
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
  
  
  /**
   * Computes class distribution for instance using decision tree.
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
  
  /**
   * Classifies a given test instance using the decision tree.
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

    return this.classifyInstance(this.m_RootNode,instance);
  }
  
  private double classifyInstance(NodeTree node, Instance instance){

    if (node.getAttribute()== null) {
      return node.getClassValue();
    } else { 
      return classifyInstance(node.getSuccesors((int) instance.value(node.getAttribute())), instance);
    }
      
  } 

  
  /**
   * Splits a dataset according to the values of a nominal attribute.
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
      return this.getStopLevel()==-1 || Utils.grOrEq(0,infoGains[m_Attribute.index()]) ||  (this.getStopLevel()>0 && this.getStopLevel()==(level));
  }
   
  /**
   * Method for building an IPTree tree.
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
      
      if (m_LeafEstimation==1){
          for (int i=0; i<data.numClasses(); i++){
            //node.getDistribution()[i]++;
            node.getFrequency()[i]++;
            //node.setSupport(node.getSupport()+1);
          }
      }
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

              if (m_LeafEstimation==1){
                  for (int i=0; i<data.numClasses(); i++){
                    //node.getSuccesors(j).getDistribution()[i]++;
                    node.getSuccesors(j).getFrequency()[i]++;
                    //node.getSuccesors(j).setSupport(node.getSupport()+1);
                  }
              }

              //node.getSuccesors(j).normalize();
              //node.getSuccesors(j).setClassValue(node.getMaxDistributionIndex());
        }else{
            makeTree(node.getSuccesors(j),splitData[j],level+1);
        }
        
      }
    }
  }


  /**
   * Computes information gain for an attribute.
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
    if (this.m_SplitMetric==IPTree.J48_INFOGAINRATIO){
        if (infoGain==0.0)
            return 0.0;
        double divinfogain= 0.0;
        for (int j = 0; j < att.numValues(); j++) {
                if (splitData[j].numInstances() > 0) {
                    divinfogain -= (splitData[j].numInstances()/(double)data.numInstances())*Utils.log2(splitData[j].numInstances()/(double)data.numInstances());
                }
        }
        return infoGain/divinfogain;
    }else
        return infoGain;
  }

  /**
   * Computes the entropy of a dataset.
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
    
    if (this.m_SplitMetric==IPTree.ENTROPY)
        return E_ContingencyTables.entropy(classCounts);
    else if (this.m_SplitMetric==IPTree.IMPRECISE_ENTROPY)
        return E_ContingencyTables.entropyImprecise(classCounts,this.getSValue());
    else if (this.m_SplitMetric==IPTree.GINI_INDEX)
        return E_ContingencyTables.entropyGiniIndex(classCounts);
    else if (this.m_SplitMetric==IPTree.J48_INFOGAINRATIO)
        return E_ContingencyTables.entropy(classCounts);
    else
        return 0.0;
        
  }
/*
  public double getFrequency(int classIndex){
      return this.m_Frequency[classIndex];
  }
  
  public double getSupport(){
      return this.m_Support;
  }
          
    public Attribute getAttribute(){
      return this.m_Attribute;
    }
  
    public IPTree[] getSuccesors(){
      return this.m_Successors;
    }
  */

/*    public double getSValue(){
        return this.m_SValue;
    }
    
    public void setSValue(double value){
        this.m_SValue=value;
    }
 */
    public SelectedTag getSplitMetric(){
        return new SelectedTag(this.m_SplitMetric, IPTree.TAGS_SPLIT_METRIC);
    }

    public void setSplitMetric(SelectedTag newmetric){
        if (newmetric.getTags() == IPTree.TAGS_SPLIT_METRIC) {
                this.m_SplitMetric = newmetric.getSelectedTag().getID();
        }
    }
    
    public boolean getMissclassifiedAllowed(){
        return this.m_MisclassifiedAllowed;
    }

    public void setMissclassifiedAllowed(boolean state){
        this.m_MisclassifiedAllowed=state;
    }
    
    public void setStopLevel(int level){
        this.m_StopLevel=level;
    }
    
    public int getStopLevel(){
        return this.m_StopLevel;
    }
    
    public SelectedTag getLeafEstimation(){
        return new SelectedTag(this.m_LeafEstimation, IPTree.TAGS_LEAF_ESTIMATION);
    }

    public void setLeafEstimation(SelectedTag newmetric){
        if (newmetric.getTags() == IPTree.TAGS_LEAF_ESTIMATION) {
                this.m_LeafEstimation = newmetric.getSelectedTag().getID();
        }
    }

    public void setKTHRootAttribute(int number){
        this.m_KThRootAttribute=number;
    }

    public int getKTHRootAttribute(){
        return this.m_KThRootAttribute;
    }

    /**
     * Return the value of the parameter 's' of the IDM model.
     * @return
     */
    public double getSValue() {
        return this.m_SValue;
    }

    /**
     * Set the value of the parameter 's' of the IDM model.
     * @param
     */
    public void setSValue(double value) {
        this.m_SValue = value;
    }

   /**
    * Gets the current settings of the filter.
    *
    * @return an array of strings suitable for passing to setOptions
    */
    public String [] getOptions() {
        String[] superOptions = super.getOptions();
        String[] options = new String[20 + superOptions.length];
        int current = 0;


        
        options[current++] = "-LeafEstimation";
        options[current++] = IPTree.STRING_LEAF_ESTIMATION[this.m_LeafEstimation];


        options[current++] = "-S"; options[current++] = ""+getSValue();
        
        options[current++] = "-StopLevel"; options[current++] = ""+ getStopLevel();
        
        options[current++] = "-SM";
        options[current++] = IPTree.STRING_SPLIT_METRIC[this.m_SplitMetric];
        
        if (this.m_MisclassifiedAllowed)
            options[current++] = "-MissClassified";
            
        options[current++] = "-KTH"; options[current++] = ""+getKTHRootAttribute();


        // insert options from parent class
        for (int iOption = 0; iOption < superOptions.length; iOption++) {
            options[current++] = superOptions[iOption];
        }

        while (current < options.length) {
          options[current++] = "";
        }
        
        return options;
        
    }
    
    
    /**
    * Parses the options for this object. Valid options are: <p>
    *
    * -R col1,col2-col4,... <br>
    * Specifies list of columns to Discretize. First
    * and last are valid indexes. (default none) <p>
    *
    * -V <br>
    * Invert matching sense.<p>
    *
    * -D <br>
    * Make binary nominal attributes. <p>
    *
    * -E <br>
    * Use better encoding of split point for MDL. <p>
    *   
    * -K <br>
    * Use Kononeko's MDL criterion. <p>
    * 
    * @param options the list of options as an array of strings
    * @exception Exception if an option is not supported
    */
    public void setOptions(String[] options) throws Exception {    
    
        String convertList = Utils.getOption("LeafEstimation",options);
        for (int i=0; i<IPTree.STRING_LEAF_ESTIMATION.length; i++){
            if (convertList.compareTo(IPTree.STRING_LEAF_ESTIMATION[i]) == 0) {
                    this.setLeafEstimation(new SelectedTag(i, IPTree.TAGS_LEAF_ESTIMATION));
                    break;
            }
        }
        
        convertList = Utils.getOption("S",options);
        if (convertList.length() != 0) {
            this.setSValue(Double.parseDouble(convertList));
        } else {
            this.setSValue(1);
        }
        
        convertList = Utils.getOption("StopLevel",options);
        if (convertList.length() != 0) {
            this.setStopLevel(Integer.parseInt(convertList));
        } else {
            this.setStopLevel(0);
        }

        convertList = Utils.getOption("SM",options);
        for (int i=0; i<IPTree.STRING_SPLIT_METRIC.length; i++){
            if (convertList.compareTo(IPTree.STRING_SPLIT_METRIC[i]) == 0) {
                    this.setSplitMetric(new SelectedTag(i, IPTree.TAGS_SPLIT_METRIC));
                    break;
            }
        }
                
        
        if (Utils.getFlag("MissClassified", options))
            this.setMissclassifiedAllowed(true);
        else
            this.setMissclassifiedAllowed(false);
        
        convertList = Utils.getOption("KTH",options);
        if (convertList.length() != 0) {
            this.setKTHRootAttribute(Integer.parseInt(convertList));
        } else {
            this.setKTHRootAttribute(1);
        }
     
        super.setOptions(options);
    }
    
   
    /**
    * Gets an enumeration describing the available options.
    *
    * @return an enumeration of all the available options.
    */
    
    public Enumeration listOptions() {

        Vector newVector = new Vector();

        
        newVector.addElement(new Option("\tLeaf Estimation Metod\n",
                  "LeafEstimation", 1,"-LeafEstimation <method>"));

        newVector.addElement(new Option(
              "\tSpecifies the level where stop the building of the tree",
              "StopLevel", 1, "-StopLevel level"));

        newVector.addElement(new Option(
              "\tSplit Metric (Entropy, IDM, GiniIndex, J48InfoGainRatio, A_NPI_M, NPI_M)\n"
              + "\t(default IDM)",
              "SM", 1, "-SM <value>"));

        newVector.addElement(new Option(
              "\tSpecifies if the missclassifications are allowed",
              "MissClassified", 0, "-MissClassified"));

        newVector.addElement(new Option(
              "\tK-th most informative attribute as root node\n"
              + "\t(default the first attribute)",
              "KTH", 1, "-KTH <type>"));


        Enumeration enu = super.listOptions();
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }

        return newVector.elements();
    }
  
  /**
   * Outputs a tree at a certain level.
   *
   * @param level the level at which the tree is to be printed
   * @return the tree as string at the given level
   */
  private String toString(NodeTree node, int level) {

    StringBuffer text = new StringBuffer();
    
    if (node.getAttribute()== null) {
      if (node.getDistribution()==null){//if (Instance.isMissingValue(node.getClassValue())) {
        text.append(": null");
      } else {
          //if (Instance.isMissingValue(node.getClassValue()))
          //    node.se
        ///text.append(": " + node.getClassAttribute().value((int) node.getClassValue())+" ("+node.getFrequency()[(int) node.getClassValue()]+"/"+node.getSupport()+") - ["+(node.getDistribution()[(int) node.getClassValue()])+"]");
        //  text.append(": " + node.getClassValue()+" ("+node.getFrequency()[(int) node.getClassValue()]+"/"+node.getSupport()+") - ["+(node.getDistribution()[(int) node.getClassValue()])+"]");
            text.append(": " + node.getClassValue()+" ("+node.getFrequency()[(int) node.getClassValue()]+"/"+node.getSupport()+") - [");
            for (int i=0; i<node.getDistribution().length; i++)
                text.append(node.getDistribution()[i]+",");
      } 
    } else {
      for (int j = 0; j < node.getAttribute().numValues(); j++) {
        text.append("\n");
        for (int i = 0; i < level; i++) {
          text.append("|  ");
        }
        text.append(node.getAttribute().name() + " = " + node.getAttribute().value(j));
        text.append(toString(node.getSuccesors(j),level + 1));
      }
    }
    return text.toString();
  }

  /**
   * Prints the decision tree using the private toString method from below.
   *
   * @return a textual description of the classifier
   */
  public String toString() {

    
    if (this.m_RootNode==null) {
      return "IPTree: No model built yet.";
    }
    return "IPTree\n\n" + toString(this.m_RootNode,0);
  }
  
  public NodeTree getRootNode(){
      return this.m_RootNode; 
  }
  

 
  

  /**
   * Main method.
   *
   * @param args the options for the classifier
   */
  public static void main(String[] args) {

    try {
      System.out.println(Evaluation.evaluateModel(new IPTree(), args));
    } catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }

}
    
