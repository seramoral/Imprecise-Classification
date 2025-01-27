/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package utils;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

/**
 *
 * @author Administrador
 */
public class WilconxonSignedRankTest {


public static int sumahasta(int x){
  int salida=0;
  for (int i=1; i < x; i++) salida=salida+i;
  return salida;
}

public static void ordenar(double v[], int tam){
  double aux;
  for (int i = 0; i < tam-1; i++){
    for (int j = 0; j <tam-i-1 ; j++){
      if (v[j] > v[j+1]){
         aux = v[j];
         v[j] = v[j+1];
         v[j+1] = aux;
      }
    }
  }
}

public static void ranking(double ranksalida[], double d[], int tam){
   double[] dord = new double[tam];
   double posicion=0;
   int cont;

   for (int i=0; i<tam; i++) dord[i]=d[i];
   ordenar(dord,tam);
   for (int k=0; k<tam; k++){
     cont=0;
     for (int j=0; j<tam; j++){
        if (d[k]==dord[j]){
          cont++;
          if (cont==1) posicion=j+1;
        }
     }

     if (cont==1) ranksalida[k]=posicion;
     else ranksalida[k]=(cont*posicion+sumahasta(cont))/(double)cont;
   }

}

public static void wilcoxon(int algorithm1, int algorithm2, double v[], double w[], int tam){
  double[] dif = new double[tam];
  double[] difabs = new double[tam];
  double[] rank = new double[tam];
  double Rmas=0.0;
  double Rmenos=0.0;
  double z=0.0;
  double T=0.0;
  int mejor=1;
  boolean iguales=true;

  for (int i=0; i<tam; i++){
    dif[i]=v[i]-w[i];
    difabs[i]=Math.abs(dif[i]);
  }
  ranking(rank, difabs, tam);

  for (int i=0; i<tam; i++){
    if (dif[i]>0) Rmas=Rmas+rank[i];
    if (dif[i]==0.0) Rmas=Rmas +(1.0/2.0)*rank[i];
  }

  for (int i=0; i<tam; i++){
    if (dif[i]<0) Rmenos=Rmenos+rank[i];
    if (dif[i]==0.0) Rmenos=Rmenos +(1.0/2.0)*rank[i];
  }

  T=Rmas;
  if (Rmas-Rmenos<0.0) mejor=2;
  if (Rmenos<Rmas) T=Rmenos;

  z=(T-(0.25*tam*(tam+1)))/Math.sqrt((1.0/24.0)*tam*(tam+1)*(2*tam+1));
  System.out.println("Valor z: "+z);
  
  
  double[] zthreshold={-2.57,-1.96,-1.64}; //Threshodls at 0.01 %, 0.05 %, 0.1 
  double[] zlevel={0.01,0.05,0.1};
  
  for (int i=0; i<zthreshold.length; i++){

      if (z<zthreshold[i]) 
          iguales=false;
      
      if (iguales==true){
            System.out.println("Algorithm "
									+ (algorithm1+1)
									+ " and "
									+ "Algorithm "
									+ (algorithm2+1)
									+ " are not significantly different when alpha="+zlevel[i]+" by Wilconxon Signed Rank Test.");          
          //System.out.println("No hay diferencias significativas");
      }else if (mejor==1) {
            System.out.println("Algorithm "
									+ (algorithm1+1)
									+ " is significantly higher than "
									+ "Algorithm "
									+ (algorithm2+1)
									+ " when alpha="+zlevel[i]+" by Wilconxon Signed Rank Test.");          

          //System.out.println("El primer clasificador es mejor");
      }else{
            System.out.println("Algorithm "          
									+ (algorithm1+1)
									+ " is significantly lower than "
									+ "Algorithm "
									+ (algorithm2+1)
									+ " when alpha=" + zlevel[i] + " by Wilconxon Signed Rank Test.");
          //System.out.println("El segundo clasificador es mejor");
      }
  }
  System.out.println("\n\n");  
//sustituir por (z<-2.57) para hacer el test al 0.01
//sustituir por (z<-1.78) para hacer el test al 0.075
//sustituir por (z<-1.64) para hacer el test al 0.1
  
  
  
  
  
  
  

}

       public static double[][] extractValues(String fileName) throws Exception{
            
            File file = new File(fileName);
            if (!file.exists()) {
                    System.out.println("data file:" + fileName + " does not exist!");
                    return null;
            }

            FileInputStream in = new FileInputStream(fileName);
            DataInputStream datain = new DataInputStream(in);

            FileInputStream in1 = new FileInputStream(fileName);
            DataInputStream datain1 = new DataInputStream(in1);

            int numOfDatasets = 1;
            int numOfAlgorithms = 0;
            String readTemp = null;
            String[] strings = null;

            readTemp = datain1.readLine();
            strings = readTemp.split(",");
            numOfAlgorithms = strings.length/2;

            while ((readTemp = datain1.readLine()) != null) {
                    numOfDatasets++;
            }

            double[][] data = new double[numOfAlgorithms][numOfDatasets];

            for (int i = 0; i < numOfDatasets; i++) {
                    readTemp = datain.readLine();
                    strings = readTemp.split(",");
                    data[0][i] = (new Double(strings[1])).doubleValue();
                    for (int j = 1; j < numOfAlgorithms; j++)
                            data[j][i] = (new Double(strings[2*j])).doubleValue();
            }
            
            return data;
                        
        }

    public static double[][] extractValues2(String fileName) throws Exception{

            File file = new File(fileName);
            if (!file.exists()) {
                    System.out.println("data file:" + fileName + " does not exist!");
                    return null;
            }

            FileInputStream in = new FileInputStream(fileName);
            DataInputStream datain = new DataInputStream(in);

            FileInputStream in1 = new FileInputStream(fileName);
            DataInputStream datain1 = new DataInputStream(in1);

            int numOfDatasets = 0;
            int numOfAlgorithms = 0;
            String readTemp = null;
            String[] strings = null;

            readTemp = datain1.readLine();
            strings = readTemp.split("\t");
            numOfAlgorithms = strings.length;

            while ((readTemp = datain1.readLine()) != null) {
                    numOfDatasets++;
            }

            double[][] data = new double[numOfAlgorithms][numOfDatasets];

            datain.readLine();
            for (int i = 0; i < numOfDatasets; i++) {
                    readTemp = datain.readLine();
                    strings = readTemp.split("\t");
                    //data[i][0] = (new Double(strings[1])).doubleValue();
                    for (int j = 0; j < numOfAlgorithms; j++)
                            data[j][i] = (new Double(strings[j])).doubleValue();
            }

            return data;

        }

public static void main(String[] args) throws Exception{

        double[][] data=WilconxonSignedRankTest.extractValues("f:/tmp/tmp.csv");
        
        for (int i=0; i<data.length-1; i++)
            for (int j=i+1; j<data.length; j++)
                WilconxonSignedRankTest.wilcoxon(i,j,data[i],data[j],data[i].length);


}    
}
