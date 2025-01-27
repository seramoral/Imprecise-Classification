/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package utils;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.classifiers.credalClassifiers.CredalClassifier;
import weka.classifiers.meta.E_FilteredClassifier;
import weka.core.Instance;

/**
 * Class to manage the credal classification statistics.
 * 
 * @author Andres Masegosa, andrew@decsai.ugr.es
 * @version $Revision: 0 $
 */
public class CredalClassificationStatistics implements Serializable{

    /** for serialization */
    static final long serialVersionUID = 2889730616939923301L;

    private Vector<Double> m_MeasureScores;
    private Vector<String> m_MeasureNames;
    private boolean m_EnumerateMeasures = true;

    private double countInstances;

    transient private FileWriter fileWriter;

    private String m_fileName;
    

    private String m_OutputFolder="./results/";

    private boolean m_OutputFlag=false;
    
    /* Matrix of costs of errors for cost-sensitive problems
        mij = cost of predicting the i-th class value when the
        real class value is the j-th value
    */ 
    private double[][] cost_errors;


    public CredalClassificationStatistics(Classifier classifier){
        m_MeasureScores = new Vector();
        m_MeasureNames = new Vector();
        m_EnumerateMeasures = true;
        countInstances=0;

        String name = classifier.getClass().getName();
        String[] options = classifier.getOptions();
        for (int i=0; i<options.length; i++)
            name+=" "+options[i];

        this.m_fileName=name.replace("-","_").replace(" ", "_").replace(".", "_").replace(":", "_").replace("\\", "_").replace("/", "_");

    }

    public void resetValues(){
        countInstances=0;
        for (int i=0; i<this.m_MeasureScores.size(); i++)
            this.m_MeasureScores.set(i,0.0);
    }
    
    public void setEnumerateMeasures(boolean enumerate) {
        this.m_EnumerateMeasures = enumerate;
    }

    public Vector<String> enumerateMeasures() {
        return this.m_MeasureNames;
    }

    public double getMeasure(String measureName) {
        int index = this.m_MeasureNames.indexOf(measureName);
        if (index != -1) {

            if (measureName.compareTo("measureSingleAccuracy")==0){
                if (this.getValue("measureDeterminacy")==0)
                    return Double.NaN;
                else
                    return this.m_MeasureScores.elementAt(index)/this.getValue("measureDeterminacy");
            }else if (measureName.compareTo("measureSetAccuracy")==0 || measureName.compareTo("measureInderminacySize")==0){
                if ((this.countInstances - this.getValue("measureDeterminacy"))==0)
                    return Double.NaN;
                else
                    return this.m_MeasureScores.elementAt(index)/(this.countInstances - this.getValue("measureDeterminacy"));
            }else{
                return this.m_MeasureScores.elementAt(index)/countInstances;
            }
        } else {
              throw new IllegalArgumentException(measureName
			  + " not supported (j48)");
        }
    }

    private double getValue(String measureName) {
        int index = this.m_MeasureNames.indexOf(measureName);
        if (index != -1) {
            return this.m_MeasureScores.elementAt(index);
        } else {
            return Double.NaN;
        }
    }

    private void setValue(String measureName, double value) {

        int index = this.m_MeasureNames.indexOf(measureName);

        this.m_MeasureScores.set(index, value);
        
    }
    
    public void updateStatistics(int[] nonDominatedClasses, Instance instance){

        updateDeterminacyStatistics(nonDominatedClasses, instance);
        updateUtilityStatistics(nonDominatedClasses, instance);
        this.countInstances++;

        if (this.m_OutputFlag)
            updateOutputFile(nonDominatedClasses,instance);
        
    }

    private int[] transform(int[] nonDominatedClasses, Instance instance){
        int[] newarray=new int[instance.classAttribute().numValues()];
        
        for (int k=0; k<newarray.length; k++){
            newarray[k]=0;
            for (int i=0; i<nonDominatedClasses.length; i++)
                if (nonDominatedClasses[i]==k){
                    newarray[k]=1;
                    break;
                }
        }

        return newarray;
        
    }
    public void updateOutputFile(int[] nonDominatedClasses, Instance instance){
        try{

            if (instance==null)
                return;

            this.fileWriter=new FileWriter(m_OutputFolder+m_fileName+".csv",true);
            String instancesName=instance.dataset().relationName();
            instancesName=instancesName.replaceAll("-weka\\.filters\\..*", "").replaceAll("-unsupervised\\..*",   "").replaceAll("-supervised\\..*",     "");
            this.fileWriter.write(instancesName+", "+instance.classValue()+", ");

            int[] array = transform(nonDominatedClasses,instance);
            for (int i=0; i<array.length-1; i++){
                    this.fileWriter.write(array[i]+", ");
            }
            this.fileWriter.write(array[array.length-1]+"\n");
            this.fileWriter.close();
        }catch(Exception ex){
            ex.printStackTrace();
        }

    }

    public void updateDeterminacyStatistics(int[] nonDominatedClasses, Instance instance){

        if (this.m_EnumerateMeasures) {
            m_MeasureNames.add("measureDeterminacy");
            m_MeasureNames.add("measureSingleAccuracy");
            m_MeasureNames.add("measureSetAccuracy");
            m_MeasureNames.add("measureInderminacySize");
            m_MeasureNames.add("measureSingleCost");
            m_MeasureNames.add("measureSetCost");

            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);

            return;
        }

        double measureDeterminacy = this.getValue("measureDeterminacy");
        double measureSingleAccuracy = this.getValue("measureSingleAccuracy");
        double measureSetAccuracy = this.getValue("measureSetAccuracy");
        double measureInderminacySize = this.getValue("measureInderminacySize");
        double measureSingleCost = this.getValue("measureSingleCost");
        double measureSetCost = this.getValue("measureSetCost");

        double cost, max_cost;
        int predicted_value, real_value;
        boolean correct = false;
        
        real_value = (int)instance.classValue();

        if (nonDominatedClasses.length==1){
            measureDeterminacy++;
            predicted_value = nonDominatedClasses[0];
            
            if (predicted_value ==real_value){
                measureSingleAccuracy++;
            }
                
            else if (cost_errors != null){
                cost = cost_errors[predicted_value][real_value];    
                measureSingleCost+=cost;
            }
        }

        else{
            measureInderminacySize+=nonDominatedClasses.length;
            
            for (int i=0; i<nonDominatedClasses.length; i++){
                if (nonDominatedClasses[i]==real_value)
                    measureSetAccuracy++;        
            }
            
            if(cost_errors != null){
                max_cost = Double.NEGATIVE_INFINITY;
                
                for (int i=0; i<nonDominatedClasses.length; i++){
                    if (nonDominatedClasses[i]==real_value){
                        correct = true;
                    }
                
                    else{
                        cost = cost_errors[nonDominatedClasses[i]][real_value];
                    
                        if(cost > max_cost)
                            max_cost = cost;
                    }
                }
            
                if(!correct)
                    measureSetCost+=max_cost;
            
                }
        }

        this.setValue("measureDeterminacy",measureDeterminacy);
        this.setValue("measureSingleAccuracy",measureSingleAccuracy);
        this.setValue("measureSetAccuracy",measureSetAccuracy);
        this.setValue("measureInderminacySize",measureInderminacySize);
        this.setValue("measureSingleCost",measureSingleCost);
        this.setValue("measureSetCost",measureSetCost);
    }
    
    /**
     * Quadratic utility u(x)=(2-4k)x^2+(4k-1)x
     * @param x argument
     * @param k value of u(0.5)
     * @return u(x;0.5)
     */
    public double quad_util(double x, double k) {
        return (2.0-4.0*k)*x*x+(4*k-1)*x;
    }
    
    /**
     * Initialize the matrix of cost errors
     * @param matrix_cost_errors the matrix of cost errors
     */
    
    public void initializeCostErrors(double[][] matrix_cost_errors){
        int num_classes = matrix_cost_errors.length;
        cost_errors = new double[num_classes][num_classes];
        
        for(int i = 0; i < num_classes; i++){
            for(int j = 0; j < num_classes; j++)
                cost_errors[i][j] = matrix_cost_errors[i][j];
        }
    }
    
    
    public void updateUtilityStatistics(int[] nonDominatedClasses, Instance instance){

        if (this.m_EnumerateMeasures) {
            m_MeasureNames.add("measureMIP.A=-1");
            m_MeasureNames.add("measureMIP.A=0");
            m_MeasureNames.add("measureMIP.A=+1");
            m_MeasureNames.add("measureMIC");
            m_MeasureNames.add("measureMIC_CostMatrix");
            m_MeasureNames.add("measureDACC"); // d-accuracy with linear utility
            m_MeasureNames.add("measureQUAD_UTIL0.6"); // d-accuracy with quadratic utility and u(0.5)=0.6
            m_MeasureNames.add("measureQUAD_UTIL0.65"); // d-accuracy with quadratic utility and u(0.5)=0.65 (matches F1-measure)
            m_MeasureNames.add("measureQUAD_UTIL0.7"); // d-accuracy with quadratic utility and u(0.5)=0.7
            m_MeasureNames.add("measureQUAD_UTIL0.75"); // d-accuracy with quadratic utility and u(0.5)=0.75
            m_MeasureNames.add("measureQUAD_UTIL0.8"); // d-accuracy with quadratic utility and u(0.5)=0.8
            
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);
            // utility measures
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);
            m_MeasureScores.add(0.0);

            return;
        }

        double measureMIP_A = this.getValue("measureMIP.A=-1");
        double measureMIP_B = this.getValue("measureMIP.A=0");
        double measureMIP_C = this.getValue("measureMIP.A=+1");
        double measureMIC = this.getValue("measureMIC");
        double measureMIC_CostMatrix = this.getValue("measureMIC_CostMatrix");

        double measureDACC = this.getValue("measureDACC");
        double measureUTIL60 = this.getValue("measureQUAD_UTIL0.6");
        double measureUTIL65 = this.getValue("measureQUAD_UTIL0.65");
        double measureUTIL70 = this.getValue("measureQUAD_UTIL0.7");
        double measureUTIL75 = this.getValue("measureQUAD_UTIL0.75");
        double measureUTIL8 = this.getValue("measureQUAD_UTIL0.8");
        
        int real_class_value = (int)instance.classValue();
        double num_class_values = (double)instance.classAttribute().numValues();
        //Compute MIP and D-ACC
        boolean rightClassification=false;
        for (int i=0; i<nonDominatedClasses.length; i++)
            if (nonDominatedClasses[i]==real_class_value){
                measureMIP_A+=-Math.log(nonDominatedClasses.length/num_class_values);
                measureMIP_B+=-Math.log(nonDominatedClasses.length/num_class_values);
                measureMIP_C+=-Math.log(nonDominatedClasses.length/num_class_values);
                measureMIC+=-Math.log(nonDominatedClasses.length/num_class_values);
                measureMIC_CostMatrix+=-Math.log(nonDominatedClasses.length/num_class_values);
                measureDACC+=1.0/(double)nonDominatedClasses.length;
                measureUTIL60+=quad_util(1.0/(double)nonDominatedClasses.length,0.6);
                measureUTIL65+=quad_util(1.0/(double)nonDominatedClasses.length,0.65);
                measureUTIL70+=quad_util(1.0/(double)nonDominatedClasses.length,0.7);
                measureUTIL75+=quad_util(1.0/(double)nonDominatedClasses.length,0.75);
                measureUTIL8+=quad_util(1.0/(double)nonDominatedClasses.length,0.8);
                rightClassification=true;
                break;
            }

        if (!rightClassification){
            double m_AlphaUtiliyMIP=-1;
            measureMIP_A+=(-1-m_AlphaUtiliyMIP)*Math.log(num_class_values);
            m_AlphaUtiliyMIP=0;
            measureMIP_B+=(-1-m_AlphaUtiliyMIP)*Math.log(num_class_values);
            m_AlphaUtiliyMIP=1;
            measureMIP_C+=(-1-m_AlphaUtiliyMIP)*Math.log(num_class_values);
            measureMIC+=(-1/(num_class_values - 1))*Math.log(num_class_values);
            
            if(cost_errors!= null){
                double max_cost_error = Double.NEGATIVE_INFINITY;
                double cost_error;
                int predicted_class_value;
         
                for (int i=0; i<nonDominatedClasses.length; i++){
                    predicted_class_value = nonDominatedClasses[i];
                 
                    cost_error = cost_errors[predicted_class_value][real_class_value];
                
                    if(cost_error > max_cost_error)
                        max_cost_error = cost_error;
                }
             
            
            
                measureMIC_CostMatrix+=(-1/(num_class_values - 1))*Math.log(num_class_values)*max_cost_error;
            }
       
        }

        this.setValue("measureMIP.A=-1",measureMIP_A);
        this.setValue("measureMIP.A=0",measureMIP_B);
        this.setValue("measureMIP.A=+1",measureMIP_C);
        this.setValue("measureMIC",measureMIC);
        this.setValue("measureMIC_CostMatrix",measureMIC_CostMatrix);
        this.setValue("measureDACC",measureDACC);
        this.setValue("measureQUAD_UTIL0.6",measureUTIL60);
        this.setValue("measureQUAD_UTIL0.65",measureUTIL65);
        this.setValue("measureQUAD_UTIL0.7",measureUTIL70);
        this.setValue("measureQUAD_UTIL0.75",measureUTIL75);
        this.setValue("measureQUAD_UTIL0.8",measureUTIL8);

    }

    public static void computeRankStatistics(String expFile) throws IOException, ClassNotFoundException, Exception{

        FileInputStream fi = new FileInputStream(expFile);
        ObjectInputStream oi = new ObjectInputStream(new BufferedInputStream(fi));
        weka.experiment.Experiment exp = (weka.experiment.Experiment)oi.readObject();
        oi.close();

        

        javax.swing.DefaultListModel listClassifier=new javax.swing.DefaultListModel();

        Classifier[] classifiers = (Classifier[]) exp.getPropertyArray();

        BufferedReader[] readers = new BufferedReader[classifiers.length];

        for (int i=0; i<classifiers.length; i++){
           // Open the file that is the first

            Classifier classifier;
            if (classifiers[i].getClass().getName().compareTo("weka.classifiers.meta.E_FilteredClassifier")==0)
                classifier=((E_FilteredClassifier)classifiers[i]).getClassifier();
            else
                classifier=classifiers[i];

            String outputFolder=null;
            try{
                if (((CredalClassifier)classifier).getOutputFlag())
                    outputFolder=((CredalClassifier)classifier).getOutputFolder();
            }catch(Exception ex){
                ex.printStackTrace();
                throw new Exception("The classifier"+ classifier.getClass().getName() +" does not" +
                        "inherits from the CredalClassifier class: ");
            }
            if (outputFolder==null)
                throw new Exception("The classifier "+ classifier.getClass().getName()+"does not properly activite the " +
                        "output predicitions.");

            String name = classifier.getClass().getName();

            String[] options = classifier.getOptions();
            for (int j=0; j<options.length; j++)
                name+=" "+options[j];
            String fileName=name.replace("-","_").replace(" ", "_").replace(".", "_").replace(":", "_").replace("\\", "_").replace("/", "_");

            FileInputStream fstream = new FileInputStream(outputFolder+fileName+".csv");
            // Get the object of DataInputStream
            DataInputStream in = new DataInputStream(fstream);
            readers[i] = new BufferedReader(new InputStreamReader(in));
        }

        Vector<double[]> scores = new Vector<double[]>();

        HashMap<String,Integer> map = new HashMap<String,Integer>();

        String[] strLine = new String[classifiers.length];
        while ((strLine[0] = readers[0].readLine()) != null)   {
            for (int i=1; i<readers.length; i++){
                strLine[i]=readers[i].readLine();
            }

            String[] piecesTmp = strLine[0].split(",");
            if (map.containsKey(piecesTmp[0])){
                map.put(piecesTmp[0], map.get(piecesTmp[0])+1);
            }else{
                map.put(piecesTmp[0],new Integer(1));
            }

            double[] singleScore = new double[classifiers.length];
            for (int i=0; i<classifiers.length; i++){
                String[] pieces = strLine[i].split(",");
                int numClasses = pieces.length - 2;
                int classValue = (int) Double.parseDouble(pieces[1]);

                int size=0;
                boolean accurate=false;
                for (int k=2; k<pieces.length; k++){
                    int value=(int) Double.parseDouble(pieces[k]);
                    int classIndex=k-2;
                    size+=value;

                    if (value==1 && classIndex==classValue)
                        accurate=true;
                        
                }

                if (accurate)
                    singleScore[i]=(double)numClasses-size+1;
                else
                    singleScore[i]=0.0;

            }

            scores.add(singleScore);
        }

        String[] datasetNames=map.keySet().toArray(new String[0]);

        int cont=0;
        for (int k=0; k<datasetNames.length; k++){

            //FileReader reader= new FileReader(((File)datasets.get(k)).getPath());
            //Instances instances = new Instances(reader);

            System.out.println("**********************************************************");
            System.out.println("*********************"+datasetNames[k]+"*************************************");
            System.out.println("**********************************************************");

            int size=map.get(datasetNames[k]);
            //size*10 pq es un 10-fold-cross repeated 10 times
            double[][] data= new double[size][classifiers.length];

            
            for (int i=0; i<size; i++)
                data[i]=scores.get(cont++);


        }

    }


    public void setOutputFolder(String name){
        this.m_OutputFolder=name;
    }

    public String getOutputFolder(){
        return this.m_OutputFolder;
    }

    public void setOutputFlag(boolean flag){
        this.m_OutputFlag=flag;
    }

    public boolean getOutputFlag(){
        return this.m_OutputFlag;
    }


    /**
     * Class to compute the rank statistics.
     * After running an experiment with several CredalClassifiers, execute
     * this class:
     *
     * CredalClassificationStatistics <exp-file>
     *
     * ALERT!!!: Files in the ouput folder should be removed before runing the experiment.
     * 
     * @param args, the first element should contain a ".exp" file.
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception{

         if (args.length==0){
             System.out.println("Error. Non Input Arguments. Usage: \n CredalClassificationStatistics <exp-file>");
             System.exit(0);
         }
         CredalClassificationStatistics.computeRankStatistics(args[0]);


    }
}