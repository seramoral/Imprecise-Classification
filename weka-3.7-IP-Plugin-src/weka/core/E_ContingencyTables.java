package weka.core;

//import com.sun.org.apache.bcel.internal.verifier.statics.DOUBLE_Upper;
import java.util.Vector;

/**
 * This class inherits from the weka.core.ContingencyTables class of weka. 
 * The added functionality is the implementing of new entropy measures 
 * for contingency tables.
 */
public class E_ContingencyTables extends ContingencyTables{

  /** for serialization */
  static final long serialVersionUID = 2889730616939923301L;
    
  static final double MINDIFF=0.00001;
    /** The natural logarithm of 2 */
  private static double log2 = Math.log(2);

  /**
   * Help method for computing entropy.
   */
  private static double lnFunc(double num){
    
    // Constant hard coded for efficiency reasons
    if (num < 1e-6) {
      return 0;
    } else {
      return num * Math.log(num);
    }
  }
  
  public static double[] reps(double[] array,int n, double svalue){

        boolean nonintegers=false;
        for (int i=0;i<n;i++){
            if (((int)(array[i]))!=array[i]){
                nonintegers=true;
                break;
            }
        }

        if (nonintegers){

            double[] pl=new double[n];
            double[] pu=new double[n];

            double total=Utils.sum(array);
            Vector<Integer> indexes=new Vector();

            for (int i=0; i<n; i++){
                pl[i]=array[i]/(total+svalue);
                pu[i]=(array[i]+svalue)/(total+svalue);

                indexes.add(new Integer(i));
            }

            return MaxEntropy(pl, pu, indexes);
        }

        double masa=1;

        double[] l=new double[n];
        System.arraycopy(array,0,l,0,n);
        double min=l[0];
        int cont=0;
        for (int i=0;i<n;i++){
            if (min>l[i]) min=l[i];
        }
        for (int i=0;i<n;i++){
            if (min==l[i]) cont++;
        }
        if (svalue<=cont)
            masa=svalue;
        else
            masa=cont;
        for (int i=0;i<n;i++){
            if (min==l[i]) l[i]=l[i]+(masa/cont);
        }
        masa=masa-cont;
        if (masa>0.0001)
            return reps(l,n,masa);
        else
            return l;

    }
  
  /**
   * Computes the imprecise entropy for s=1 of the given array.
   *
   * @param array the array
   * @return the entropy
   */
  public static double entropyGiniIndex(double[] array) {
  
    double gini = 0;
    double total=0;
    for (int j = 0; j < array.length; j++) {
        total+=array[j];
    }
    for (int j = 0; j < array.length; j++) {
      if (array[j] > 0) {
        gini -= (array[j]/total) * (array[j]/total);
      }
    }
   
    return gini;
  
  }  
  
  /**
   * Computes the imprecise entropy for s=1 of the given array.
   *
   * @param array the array
   * @return the entropy
   */
  public static double entropyImprecise(double[] array, double svalue) {

    double returnValue = 0, sum = 0;
    double[] arraynew=reps(array,array.length, svalue);
    
    for (int i = 0; i < arraynew.length; i++) {
      if (arraynew[i]>0)
        returnValue -= arraynew[i]*Utils.log2(arraynew[i]);
      sum += arraynew[i];
    }

    if (Utils.eq(sum, 0)) {
      return 0;
    } else {
      returnValue/=sum;
      return (returnValue + Utils.log2(sum));
    }
  }
 

  
    
  /**
   * Computes the imprecise entropy for s=1 of the given array.
   *
   * @param array the array
   * @return the entropy
   */
  public static double entropyNPI(double[] array) {

    double returnValue = 0;
    double[] arraynew=NPITransformation(array);
    
   
    for (int i = 0; i < arraynew.length; i++) {
      if (arraynew[i]>0)
        returnValue -= arraynew[i]*Utils.log2(arraynew[i]);
    }
    
    return returnValue;
    
    /*
    if (Utils.eq(sum, 0)) {
      return 0;
    } else {
      returnValue/=sum;
      return (returnValue + Utils.log2(sum));
    }
     */
  }
 
 public static double[] NPITransformation(double[] array){
      
      double[] pl=new double[array.length];
      double[] pu=new double[array.length];
      double sum=Utils.sum(array);
      
      for (int i=0; i<array.length; i++){
          pl[i]=Math.max(0,(array[i]-1)/sum);
          pu[i]=Math.min(1,(array[i]+1)/sum);
      }
      Vector<Integer> indexes=new Vector();
      for (int i=0; i<array.length; i++){
          indexes.add(new Integer(i));
      }
      
      try{
        return MaxEntropy(pl,pu,indexes);
      }catch(Exception e){
          System.out.println("D");
          return null;
      }
  }
      
private static double[] MaxEntropy(double pl[], double pu[], Vector<Integer> indexes){
    
    

     if (Math.abs(1-Utils.sum(pl))<0.001){ //if ((Suma(pl,m)<1.00001)&&(Suma(pl,m)>1.0-0.00001))   //SUMA UNO
       double[] pmax=new double[pl.length];
       System.arraycopy(pl,0,pmax,0,pl.length);
       return pmax;
     } else {
        
        for (int i=0;i<indexes.size();i++){
            if (Math.abs(pl[indexes.elementAt(i).intValue()]-pu[indexes.elementAt(i).intValue()])<MINDIFF)
                indexes.remove(i);
        }

        double sum=Utils.sum(pl);
        int men=minIndex(pl,indexes);
        int sig=Siguiente(pl,men,indexes);
        int nme=Nmenores(pl,men,indexes);
        
        double val=pl[men];
        
        for (int i=0;i<pl.length;i++){
               if (Math.abs(pl[i]-val)<MINDIFF){  //if ((pl[i]<val+0.00001)&&(pl[i]>val-0.00001)) {
                 if (!(sig==men)){
                    pl[i]=pl[i]+Math.min(Math.min(pu[i]-pl[i],(pl[sig]-val)),(1.0-sum)/(double)nme);
                 }else{
                    pl[i]=pl[i]+Math.min(Math.min(pu[i]-pl[i],(1.0-sum)/(double)nme),1);                       
                 }
              }
        }
        return MaxEntropy(pl,pu,indexes);
      }
}       

private static int minIndex(double p[], Vector<Integer> indexes){
    double min=Double.MAX_VALUE;
    int index=-1;
    for (int i=0; i<indexes.size(); i++)
        if (p[indexes.elementAt(i).intValue()]<min){
            min=p[indexes.elementAt(i).intValue()];
            index=indexes.elementAt(i).intValue();
        }
            
    return index;
}
 private static int Nmenores(double p[], int indexmenor, Vector<Integer> indexes){
    int nmenores=0;
    for (int i=0; i<indexes.size(); i++){
      if (Math.abs(p[indexes.elementAt(i).intValue()]-p[indexmenor])<MINDIFF)//if ((p[i]>p[Menor(p,m)]-0.0000001)&&(p[i]<p[Menor(p,m)]+0.0000001)) 
          nmenores++;
    }
    return nmenores;
 }

private static int Siguiente(double p[],int indexmenor, Vector<Integer> indexes){
   double[] q = new double[indexes.size()];
   boolean different=false;
  for (int i=0; i<indexes.size(); i++){
     if (Math.abs(p[indexes.elementAt(i).intValue()]-p[indexmenor])<MINDIFF){ //if (p[i]<p[Menor(p,m)]+0.00000001)
        q[i]=Double.MAX_VALUE;
     }else{
        different=true;
        q[i]=p[indexes.elementAt(i).intValue()];
     }
  }
  if (different)
    return indexes.elementAt(Utils.minIndex(q)).intValue();
  else
    return indexmenor;
 }
      
  
  /**
   * Computes the entropy of the given array.
   *
   * @param array the array
   * @return the entropy
   */
  public static double entropyLaplace(double[] array) {

    double returnValue = 0, sum = 0;

    for (int i = 0; i < array.length; i++) {
      returnValue -= lnFunc(array[i]+1);
      sum += array[i];
    }
    if (Utils.eq(sum, 0)) {
      return 0;
    } else {
      return (returnValue + lnFunc(sum)) / (sum * log2);
    }
  }

   /**
   * Computes the entropy of the given array.
   *
   * @param array the array
   * @return the entropy
   */
  public static double entropyL1O(double[] array) {

    double returnValue = 0, sum = 0;

    for (int i = 0; i < array.length; i++) {
      returnValue -= lnFunc(array[i]);
      sum += array[i];
    }
    sum+=array.length-1;
    if (Utils.eq(sum, 0)) {
      return 0;
    } else {
      return (returnValue + lnFunc(sum)) / (sum * log2);
    }
  }
 
  
  
  
  /**
   * Computes the columns' entropy for the given contingency table.
   *
   * @param matrix the contingency table
   * @return the columns' entropy
   */
  public static double impreciseentropyOverColumns(double[][] matrix, double svalue){
    
    double returnValue = 0, sumForColumn, total = 0;
    double[] I_matrix=new double[matrix[0].length];
    
    
    for (int j = 0; j < matrix[0].length; j++){
      I_matrix[j] = 0;
      for (int i = 0; i < matrix.length; i++) {
	I_matrix[j] += matrix[i][j];
      }
    }

    return entropyImprecise(I_matrix,svalue);
    
  }
/**
   * Computes conditional entropy of the columns given
   * the rows.
   *
   * @param matrix the contingency table
   * @return the conditional entropy of the columns given the rows
   */
  public static double impreciseentropyConditionedOnRows(double[][] matrix,double svalue) {
    
    double returnValue = 0, total = 0;
    double[][] I_matrix=new double[matrix.length][matrix[0].length];
    
    for (int i=0; i<matrix.length; i++)
       I_matrix[i]=reps(matrix[i],matrix[0].length,svalue);
   
    for (int i = 0; i < matrix.length; i++) {
      double sumForRow = 0;
      double sumExactForRow = 0;
      double entropyForRow = 0;


      for (int j = 0; j < matrix[0].length; j++) {
            entropyForRow = entropyForRow + lnFunc(I_matrix[i][j]);
            sumForRow += I_matrix[i][j];

            sumExactForRow+=matrix[i][j];
            total+=matrix[i][j];
      }
      entropyForRow = (entropyForRow - lnFunc(sumForRow))/sumForRow;

      returnValue += sumExactForRow*entropyForRow;
      
    }
    if (Utils.eq(total, 0)) {
      return 0;
    }
    return -returnValue / (total * log2);
  }


  
  /**
   * Computes the columns' entropy for the given contingency table.
   *
   * @param matrix the contingency table
   * @return the columns' entropy
   */
  public static double NPIEntropyOverColumns(double[][] matrix){
    
    double[] I_matrix=new double[matrix[0].length];
    
    
    for (int j = 0; j < matrix[0].length; j++){
      I_matrix[j] = 0;
      for (int i = 0; i < matrix.length; i++) {
	I_matrix[j] += matrix[i][j];
      }
    }

    return entropyNPI(I_matrix);
  }
/**
   * Computes conditional entropy of the columns given
   * the rows.
   *
   * @param matrix the contingency table
   * @return the conditional entropy of the columns given the rows
   */
  public static double NPIEntropyConditionedOnRows(double[][] matrix) {
    
    double returnValue = 0, total = 0;
    double[][] I_matrix=new double[matrix.length][matrix[0].length];
    
    for (int i=0; i<matrix.length; i++)
       I_matrix[i]=NPITransformation(matrix[i]);
   
    for (int i = 0; i < matrix.length; i++) {
      double sumForRow = 0;
      double sumExactForRow = 0;
      double entropyForRow = 0;


      for (int j = 0; j < matrix[0].length; j++) {
            entropyForRow = entropyForRow + lnFunc(I_matrix[i][j]);
            sumForRow += I_matrix[i][j];

            sumExactForRow+=matrix[i][j];
            total+=matrix[i][j];
      }
      entropyForRow = (entropyForRow - lnFunc(sumForRow))/sumForRow;

      returnValue += sumExactForRow*entropyForRow;

    }
    if (Utils.eq(total, 0)) {
      return 0;
    }
    return -returnValue / (total * log2);
  }
  
  /**
   * Main method for testing this class.
   */
  public static void main(String[] ops) {

    //double[] pl = {0.0, 0.3, 0.1, 0.1, 0.0};
    //double[] pu = {0.3, 0.5, 0.5, 0.4, 0.1};

      /*
    double[] pl = {0.0, 0.0, 0.0, 0.0, 0.0};
    double[] pu = {1.0, 1.0, 1.0, 1.0, 0.0001};

    Vector<Integer> indexes=new Vector();
    for (int i=0; i<pl.length; i++){
        indexes.add(new Integer(i));
    }
    
    double[] pmax = MaxEntropy(pl,pu,indexes);
    
    for (int i=0; i<pmax.length; i++){
        System.out.print(pmax[i]+"\t");
    }
    */


    /*

        double[] dist={2,0.6,0.4};

    dist=E_ContingencyTables.reps(dist, dist.length, 1);

    for (int i=0; i<dist.length; i++)
        System.out.print(dist[i]+",");
*/

      /*
    double[] pl = {2.0/4.0, 0.6/4.0, 0.4/4.0};
    double[] pu = {3.0/4.0, 1.6/4.0, 1.4/4.0};

    Vector<Integer> indexes=new Vector();
    for (int i=0; i<pl.length; i++){
        indexes.add(new Integer(i));
    }

    double[] pmax = MaxEntropy(pl,pu,indexes);

    for (int i=0; i<pmax.length; i++){
        System.out.print(pmax[i]+"\t");
    }
*/

      //double[][] dist = {{0.010204449402759806, 2.131791585001831}, {0.03385018932347067, 0.0001620339186317366}};
      double svalue=1;
      double[][] dist = {{10, 3}, {4, 0}};

      //double[][] dist = {{2, 1}, {1, 1}, {1,0}, {1,0}, {1,1}, {1,0}};

      //System.out.println(E_ContingencyTables.impreciseentropyOverColumns(dist, 1));
      //System.out.println(E_ContingencyTables.impreciseentropyConditionedOnRows(dist,1));

      System.out.println("Info-Gain: "+(ContingencyTables.entropyOverColumns(dist)-ContingencyTables.entropyConditionedOnRows(dist)));
      System.out.println("Info-Gain Ratio: "+(ContingencyTables.entropyOverColumns(dist)-ContingencyTables.entropyConditionedOnRows(dist))/ContingencyTables.entropyOverRows(dist));

      System.out.println("Imprecise Info-Gain: "+(E_ContingencyTables.impreciseentropyOverColumns(dist, svalue)-E_ContingencyTables.impreciseentropyConditionedOnRows(dist,svalue)));



      double[] cx=new double[4];//{0.010204449402759806, 2.131791585001831,0.03385018932347067, 0.0001620339186317366};
      double[] c= new double[2];//{0.010204449402759806+0.03385018932347067,2.131791585001831+0.0001620339186317366};
      double[] x= new double[2];//{0.010204449402759806+2.131791585001831,0.03385018932347067+0.0001620339186317366};

      cx[0]=dist[0][0];
      cx[1]=dist[0][1];
      cx[2]=dist[1][0];
      cx[3]=dist[1][1];

      c[0]=dist[0][0]+dist[1][0];
      c[1]=dist[0][1]+dist[1][1];

      x[0]=dist[0][0]+dist[0][1];
      x[1]=dist[1][0]+dist[1][1];
      //System.out.println(E_ContingencyTables.entropyImprecise(cx, 1));
      //System.out.println(E_ContingencyTables.entropyImprecise(c, 1));
      //System.out.println(E_ContingencyTables.entropyImprecise(x, 1));

      System.out.println(E_ContingencyTables.entropyImprecise(c, svalue)+E_ContingencyTables.entropyImprecise(x, svalue)-E_ContingencyTables.entropyImprecise(cx, svalue));
  }

}








