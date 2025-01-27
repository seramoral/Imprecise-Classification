/*
 * BiasVarianceAnalysis.java
 *
 * Created on 3 de julio de 2006, 15:40
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package utils;

import weka.experiment.Experiment;
import weka.core.*;
import weka.core.OptionHandler;
import weka.classifiers.*;

import java.io.*;
import java.util.*;



import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.Option;
import weka.core.Instances;
import weka.core.FastVector;
import weka.core.xml.KOML;
import weka.core.xml.XMLOptions;
import weka.experiment.xml.XMLExperiment;

import java.io.Serializable;
import java.io.File;
import java.util.Enumeration;
import javax.swing.DefaultListModel;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.BufferedInputStream;



/**
 *
 * @author Andres
 */
public class BiasVarianceAnalysis implements Serializable, OptionHandler{
    
    private BVDecomposeSegCVSub bias;
    private BVDecompose biasKW;
    private Experiment exp=null;
    private String m_output;
    /** Creates a new instance of BiasVarianceAnalysis */
    public BiasVarianceAnalysis() {
        
    }

    
    /**
    * Gets the current settings of the experiment iterator.
    *
    * @return an array of strings suitable for passing to setOptions
    */
    public String [] getOptions() {
        
        return bias.getOptions();
    }

    public void setOptions(String [] options) throws Exception {
        //bias.setOptions(options);
    }

    public Enumeration listOptions() {
        return null;//bias.listOptions();
    }
    
    public void setExperiment(Experiment experiment){
        this.exp=experiment;
    }
    
    public void setOutput(String file){
        this.m_output=file;
    }

    /**
    * Runs all iterations of the experiment, continuing past errors.
    */
    public void runExperiment() throws Exception{
        
        DefaultListModel databases=exp.getDatasets();
        Classifier[] classifiers=(Classifier[])exp.getPropertyArray();
        
        double[][][] results=new double[databases.size()][classifiers.length][5];
        for (int i=0; i<databases.size(); i++){
            for (int j=0; j<classifiers.length; j++){
                bias= new BVDecomposeSegCVSub();
                bias.setClassIndex(0);
                bias.setClassifyIterations(10);
                bias.setP(-1);
                bias.setSeed(1);
                bias.setTrainSize(-1);
                bias.setDataFileName(((File)databases.elementAt(i)).getAbsolutePath());
                bias.setClassifier(classifiers[j]);
                try{
                    bias.decompose();
                    System.out.println(bias.toString());
                    int cont=0;
                    results[i][j][cont++]=bias.getError();
                    results[i][j][cont++]=bias.getKWBias();
                    results[i][j][cont++]=bias.getKWVariance();
                    results[i][j][cont++]=bias.getWBias();
                    results[i][j][cont++]=bias.getWVariance();
                }catch(Exception e){
                    int cont=0;
                    results[i][j][cont++]=Double.NaN;
                    results[i][j][cont++]=Double.NaN;
                    results[i][j][cont++]=Double.NaN;
                    results[i][j][cont++]=Double.NaN;
                    results[i][j][cont++]=Double.NaN;
                }
            }
        }
        
        
        System.out.println();
        System.out.println();
        System.out.println();
        
        FastVector attInfo=new FastVector(7);
        
        FastVector states = new FastVector(databases.size());        
        for (int i=0; i<databases.size(); i++){
           states.addElement(((File)databases.elementAt(i)).getName());
        }
        attInfo.addElement(new Attribute("DataBase",states));
        

        states = new FastVector(classifiers.length);        
        for (int i=0; i<classifiers.length; i++){
            String state=classifiers[i].getClass().getName()+" ";
            String[] options=classifiers[i].getOptions();
            for (int m=0; m<options.length; m++)
                state+=options[m]+" ";
            
           states.addElement(state);
        }
        attInfo.addElement(new Attribute("Classifier",states));

        attInfo.addElement(new Attribute("Error"));
        attInfo.addElement(new Attribute("BiasK"));
        attInfo.addElement(new Attribute("VarianceK"));
        attInfo.addElement(new Attribute("BiasW"));
        attInfo.addElement(new Attribute("VarianceW"));
        Instances instances=new Instances("BIAS_VARIANCE",attInfo,0);
        
        
        for (int i=0; i<databases.size(); i++){
            for (int j=0; j<classifiers.length; j++){
                Instance instance=new Instance(instances.numAttributes());
                instance.setValue(0,i);
                instance.setValue(1,j);
                for (int k=0; k<5; k++){
                    instance.setValue(k+2,results[i][j][k]);
                }
                instance.setDataset(instances);
                instances.add(instance);
            }
        }
        instances.compactify();        

        FileWriter fw= new FileWriter(this.m_output);
        fw.write(instances.toString());
        fw.close();
        
        
        System.out.println("Results: ");
        
        System.out.print("DataBase\tClassifier\tError\tBiasK\tVarianceK\tBiasW\tVarianceW");
        System.out.println();
        for (int i=0; i<databases.size(); i++){
            for (int j=0; j<classifiers.length; j++){
                System.out.print(((File)databases.elementAt(i)).getName()+"\t");
                System.out.print(classifiers[j].getClass().getName()+" ");
                String[] options=classifiers[j].getOptions();
                for (int m=0; m<options.length; m++)
                    System.out.print(options[m]+" ");
                System.out.print("\t");
                for (int k=0; k<5; k++){
                    System.out.print(results[i][j][k]+"\t");
                }
                System.out.println();
            }
        }
        
    }    




    /**
    * Runs all iterations of the experiment, continuing past errors.
    */
    public void runExperimentKW() throws Exception{

        DefaultListModel databases=exp.getDatasets();
        Classifier[] classifiers=(Classifier[])exp.getPropertyArray();

        double[][][] results=new double[databases.size()][classifiers.length][5];
        for (int i=0; i<databases.size(); i++){
            for (int j=0; j<classifiers.length; j++){
                biasKW= new BVDecompose();
                biasKW.setClassIndex(0);
                //biasKW.setClassifyIterations(10);
                //biasKW.setP(-1);
                biasKW.setSeed(1);
                //biasKW.setTrainSize(-1);
                biasKW.setDataFileName(((File)databases.elementAt(i)).getAbsolutePath());
                biasKW.setClassifier(classifiers[j]);
                try{
                    biasKW.decompose();
                    System.out.println(biasKW.toString());
                    int cont=0;
                    results[i][j][cont++]=biasKW.getError();
                    results[i][j][cont++]=biasKW.getBias();
                    results[i][j][cont++]=biasKW.getVariance();
                    results[i][j][cont++]=0;
                    results[i][j][cont++]=0;
                }catch(Exception e){
                    int cont=0;
                    results[i][j][cont++]=Double.NaN;
                    results[i][j][cont++]=Double.NaN;
                    results[i][j][cont++]=Double.NaN;
                    results[i][j][cont++]=Double.NaN;
                    results[i][j][cont++]=Double.NaN;
                }
            }
        }


        System.out.println();
        System.out.println();
        System.out.println();

        FastVector attInfo=new FastVector(7);

        FastVector states = new FastVector(databases.size());
        for (int i=0; i<databases.size(); i++){
           states.addElement(((File)databases.elementAt(i)).getName());
        }
        attInfo.addElement(new Attribute("DataBase",states));


        states = new FastVector(classifiers.length);
        for (int i=0; i<classifiers.length; i++){
            String state=classifiers[i].getClass().getName()+" ";
            String[] options=classifiers[i].getOptions();
            for (int m=0; m<options.length; m++)
                state+=options[m]+" ";

           states.addElement(state);
        }
        attInfo.addElement(new Attribute("Classifier",states));

        attInfo.addElement(new Attribute("Error"));
        attInfo.addElement(new Attribute("BiasK"));
        attInfo.addElement(new Attribute("VarianceK"));
        attInfo.addElement(new Attribute("BiasW"));
        attInfo.addElement(new Attribute("VarianceW"));
        Instances instances=new Instances("BIAS_VARIANCE",attInfo,0);


        for (int i=0; i<databases.size(); i++){
            for (int j=0; j<classifiers.length; j++){
                Instance instance=new Instance(instances.numAttributes());
                instance.setValue(0,i);
                instance.setValue(1,j);
                for (int k=0; k<5; k++){
                    instance.setValue(k+2,results[i][j][k]);
                }
                instance.setDataset(instances);
                instances.add(instance);
            }
        }
        instances.compactify();

        FileWriter fw= new FileWriter(this.m_output);
        fw.write(instances.toString());
        fw.close();


        System.out.println("Results: ");

        System.out.print("DataBase\tClassifier\tError\tBiasK\tVarianceK\tBiasW\tVarianceW");
        System.out.println();
        for (int i=0; i<databases.size(); i++){
            for (int j=0; j<classifiers.length; j++){
                System.out.print(((File)databases.elementAt(i)).getName()+"\t");
                System.out.print(classifiers[j].getClass().getName()+" ");
                String[] options=classifiers[j].getOptions();
                for (int m=0; m<options.length; m++)
                    System.out.print(options[m]+" ");
                System.out.print("\t");
                for (int k=0; k<5; k++){
                    System.out.print(results[i][j][k]+"\t");
                }
                System.out.println();
            }
        }

    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception{
        // TODO code application logic here
/*
        args=new String[4];
        args[0]="-E";
        args[1]="F:/andres/elvira/joaquin/experimentos3/combinedTrees/BT-EVALUATION-03.exp";
        args[2]="-O";
        args[3]="F:/andres/elvira/joaquin/experimentos3/combinedTrees/BT-EVALUATION-03.arff";
*/
        try {
            BiasVarianceAnalysis bias = null;
            Experiment exp=null;
            // get options from XML?
            String xmlOption = Utils.getOption("xml", args);
            if (!xmlOption.equals(""))
                args = new XMLOptions(xmlOption).toArray();

            String expFile = Utils.getOption('E', args);

            if (expFile.length() == 0) {
                throw new Exception("Required: -E <exp file>");
            }
            String outFile = Utils.getOption('O', args);

            if (outFile.length() == 0) {
                throw new Exception("Required: -O <outFile file>");
            }

            bias = new BiasVarianceAnalysis();
            try {
                bias.setOptions(args);
                Utils.checkForRemainingOptions(args);
            } catch (Exception ex) {
                String result = "Usage:\n\n"
                + "-E <exp|xml file>\n"
                + "\tLoad experiment from file (default use cli options).\n"
                + "\tThe type is determined, based on the extension (" 
                + " .exp or .xml)\n"
                + "\n";
                Enumeration enm = ((OptionHandler)bias).listOptions();
                while (enm.hasMoreElements()) {
                    Option option = (Option) enm.nextElement();
                    result += option.synopsis() + "\n";
                    result += option.description() + "\n";
                }
                throw new Exception(result + "\n" + ex.getMessage());
            }
            // KOML?
            if ( (KOML.isPresent()) && (expFile.toLowerCase().endsWith(KOML.FILE_EXTENSION)) ) {
                exp = (Experiment) KOML.read(expFile);
            }
            else
            // XML?
            if (expFile.toLowerCase().endsWith(".xml")) {
                XMLExperiment xml = new XMLExperiment(); 
                exp = (Experiment) xml.read(expFile);
            }
            // binary
            else {
                FileInputStream fi = new FileInputStream(expFile);
                ObjectInputStream oi = new ObjectInputStream(
                                       new BufferedInputStream(fi));
                exp = (Experiment)oi.readObject();
                oi.close();
            }

            System.err.println("Experiment:\n" + exp.toString());

            bias.setExperiment(exp);
            bias.setOutput(outFile);
            bias.runExperiment();


            /*
            if (runExp) {
            System.err.println("Initializing...");
            exp.initialize();
            System.err.println("Iterating...");
            exp.runExperiment();
            System.err.println("Postprocessing...");
            exp.postProcess();
            }
            */
        } catch (Exception ex) {
            ex.printStackTrace();
            System.err.println(ex.getMessage());
        }
    }
    
}
