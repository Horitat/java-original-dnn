package util;

/**
 * 活性化関数を定義するクラス
 *
 * */
public class ActivationFunction {
	/**ステップ関数*/
	public static int step_function(float x){
		if(x > 0.f){
			return 1;
		}
		return 0;
	}


	/**
	 * ロジスティックシグモイド
	 * @param x 入力
	 * @return シグモイド関数で活性化した結果
	 */
	public static float logistic_sigmoid(float x){
		return (float) (1. / (1. + Math.exp(-x)));
	}
	/**
	 *ロジスティックシグモイドの微分
	 * @param y 出力
	 * @return ロジスティックシグモイドの微分
	 */
	public static float dsigmoid(float y) {
		return (float) (y * (1. - y));
	}

	/**
	 * ロジスティックシグモイド
	 * @param x 入力
	 * @return シグモイド関数で活性化した結果
	 */
	public static double logistic_sigmoid(double x){
		return (double) (1. / (1. + Math.exp(-x)));
	}
	/**
	 *ロジスティックシグモイドの微分
	 * @param y 出力
	 * @return ロジスティックシグモイドの微分
	 */
	public static double dsigmoid(double y) {
		return (double) (y * (1. - y));
	}

	/**
	 * シンプルシグモイド
	 * @param x 入力
	 * @return 活性化した結果
	 */
	public static float simple_sigmoid(float x){
		return 1.f / (1.f + x);
	}
	/**
	 *シンプルシグモイドの微分
	 * @param y 出力
	 * @return シンプルシグモイドの微分
	 */
	public static float dsimple_sigmoid(float y) {
		return (float) ((-(1. + y))*(-(1. + y)));

	}

	/**ハイパーボリックタンジェント*/
	public static float tanh(float x) {
		return (float) Math.tanh(x);
	}
	/**ハイパーボリックタンジェントの微分*/
	public static float dtanh(float y) {
		return 1.f - y * y;
	}

	/**ハイパーボリックタンジェント*/
	public static double tanh(double x) {
		return (double) Math.tanh(x);
	}

	/**ハイパーボリックタンジェントの微分*/
	public static double dtanh(double y) {
		return 1. - y * y;
	}

	/**ReLU関数*/
	public static float ReLU(float x) {
		if(x > 0) {
			//System.out.println("relu");
			return x;
		} else {
			return 0.f;
		}
	}
	/**ReLUの微分*/
	public static float dReLU(float y) {
		if(y > 0) {
			//System.out.println("drelu");
			return 1.f;
		} else {
			return 0.f;
		}
	}

	/**ReLU関数*/
	public static double ReLU(double x) {
		if(x > 0) {
			//System.out.println("relu");
			return x;
		} else {
			return 0.;
		}
	}
	/**ReLUの微分*/
	public static double dReLU(double y) {
		if(y > 0) {
			//System.out.println("drelu");
			return 1.;
		} else {
			return 0.;

		}
	}




	/**ソフトマックス関数*/
	public static float[] softmax(float[] activation, int output_N) {
		// TODO 自動生成されたメソッド・スタブ
		float[] out = new float[output_N];
		float max = 0.f;
		float sum = 0.f;
		for(float n : activation){
			if(max < n){
				max = n;
			}
			//System.out.println("act:"+n);
		}

		for(int i=0; i<output_N; i++){
			out[i] = (float) Math.exp(activation[i] - max);
			sum += out[i];
		}

		for(int i=0; i<output_N; i++){
			out[i] = out[i] / sum;
		}

		return out;
	}

	public static double[] softmax(double[] activation, int output_N) {
		double[] out = new double[output_N];
		double max = -100;
		double sum = 0;
		for(double n : activation){
			if(max < n){
				max = n;
			}
			//System.out.println("act:"+n);
		}

		for(int i=0; i<output_N; i++){
			out[i] = (float) Math.exp(activation[i] - max);
			sum += out[i];
		}

		for(int i=0; i<output_N; i++){
			out[i] = out[i] / sum;
		}

		return out;
	}





	public static void main(String[] args) {
		float[] aa = {0.0f, 500.25f, 100f, 200f, 10f, -10.25f, -50.2f};
		//int[] b = {5000,1,2,3,4,5};
		//System.out.println (b);
		for(int i = 0; i< aa.length; i++);
		//System.out.println (aa[i]);
		System.out.println (softmax(aa, 3));
	}
}
