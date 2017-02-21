package test_ujmp;

import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class Sample_ulmp {

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		Matrix x = DenseMatrix.Factory.linkToArray(
				new double[] {1, 2},
				new double[] {3, 4}
				);

		Matrix y = DenseMatrix.Factory.linkToArray(
				new double[] {5, 6},
				new double[] {7, 8}
				);
		// (a)
		System.out.println( x.plus(y) );

		System.out.println("-----");
		// (b)
		System.out.println( x.mtimes(y) );

		System.out.println("-----");
		// (c)
		System.out.println( x.transpose() );
	}

}
