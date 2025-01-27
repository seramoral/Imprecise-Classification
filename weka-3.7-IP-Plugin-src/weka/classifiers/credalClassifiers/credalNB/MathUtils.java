package weka.classifiers.credalClassifiers.credalNB;

public final class MathUtils {

	// Evaluate n!
    static long factorial( int n )
    {
        if( n <= 1 )     // base case
            return 1;
        else
            return n * factorial( n - 1 );
    }
   
}
