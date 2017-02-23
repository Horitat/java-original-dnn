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

		DoubleMatrix v = new DoubleMatrix(new double[][] {
				{5, 6},
				{7, 8},
				{9,10},
				{11,12}
		});

		// (a)
		System.out.println("x+y");
		System.out.println( x.add(y) );

		System.out.println("xy");
		// (b)
		System.out.println( x.mmul(y) );
		System.out.println("x*y");
		// (b)
		System.out.println( x.mul(y) );
		System.out.println("x^t");
		// (c)
		System.out.println( x.transpose() );
		System.out.println("y(1,4)=14");
		y.put(3, 14);
		System.out.println( y );
		System.out.println("v-1");
		System.out.println( v.sub(1) );
		System.out.println("v");
		System.out.println( v );
		System.out.println("func");
		DoubleMatrix f = test(x,y);
		System.out.println( x );
//		System.out.println( x.subColumnVector(DoubleMatrix.ones(2)) );
		System.out.println("func");
		//y= y.add(5);
		System.out.println( y );
		System.out.println("end");
	}

	static DoubleMatrix test(DoubleMatrix p, DoubleMatrix k){
		p.addi(k);
		System.out.println("p");
		//y= y.add(5);
		System.out.println( p );
		k.put(0, 110);
		return p;
	}
}
