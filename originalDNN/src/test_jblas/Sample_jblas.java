package test_jblas;

import org.jblas.DoubleMatrix;

public class Sample_jblas {

	public static void main(String[] args) {
		DoubleMatrix x = new DoubleMatrix(new double[][] {
				{1, 2},
				{3, 4}
		});

		DoubleMatrix y = new DoubleMatrix(new double[][] {
				{5, 6},
				{7, 8}
		});
		// (a)
		System.out.println( x.add(y) );
		y.put(0,0, 14);
		System.out.println("-----");
		// (b)
		System.out.println( x.mmul(y) );

		System.out.println("-----");
		// (c)
		System.out.println( x.transpose() );

		System.out.println( y );
	}
}
