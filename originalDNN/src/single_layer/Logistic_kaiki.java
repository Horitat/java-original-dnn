package single_layer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import util.ActivationFunction;
import util.Common_method;
import util.GaussianDistribution;
import Mersenne.Sfmt;

public class Logistic_kaiki {
	public float[][] weight;
	public float[] bias;
	public int input_N;
	public int output_N;

	public Logistic_kaiki(int input, int output){
		input_N = input;
		output_N = output;

		weight = new float[output_N][input_N];
		bias = new float[output_N];
	}

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ

		int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
		Sfmt mt = new Sfmt(init_key);

		int input_N = 2;//入力の数
		final int classes = 3;
		final int output_N = classes;
		final int train_N = 500 * classes;//学習データの数
		final int test_N = 100 * classes;//テストデータの数

		final int epochs = 2000;//トレーニングの最大世代数
		float l_rate = 1.0f;//学習率

		int minibatchsize = 50; //ミニバッチサイズ
		int minibatch_N = train_N / minibatchsize;//ミニバッチの数

		float[][][] train_minibatch = new float[minibatch_N][minibatchsize][input_N];//学習の入力データ
		//FloatMatrix train_output = new FloatMatrix(new float[minibatch_N][minibatchsize][input_N]);
		int[][][] train_minibatch_label = new int[minibatch_N][minibatchsize][input_N];//学習のラベル

		List<Integer> minibatchindex = new ArrayList<>(); //SGDを適用する順番
		//学習データをシャッフルするための番号
		for(int i=0; i<train_N; i++){
			minibatchindex.add(i);
		}
		//System.out.println(minibatchindex);
		Common_method.list_shuffle(minibatchindex, mt);
		//Collections.shuffle(minibatchindex);
		//System.out.println("--------------------------------------------");
		//System.out.println(minibatchindex);



		float[][] traindata = new float[train_N][input_N];
		int[][] trainlabel = new int[train_N][classes];

		float[][] testdata = new float[test_N][input_N];
		Integer[][] testlabel = new Integer[test_N][classes];

		Integer[][] predict = new Integer[test_N][classes];

		GaussianDistribution g1 = new GaussianDistribution(-2., 1., mt);
		GaussianDistribution g2 = new GaussianDistribution(2., 1., mt);
		GaussianDistribution g3 = new GaussianDistribution(0.0, 1., mt);

		//データの生成
		for(int i=0; i<train_N; i++){
			if(i < train_N / classes){
				//				for(int j=0; j < input_N; j++){
				//					traindata[i][j] = (float) g1.random();
				//				}
				traindata[i][0] = (float)g1.random();
				traindata[i][1] = (float)g2.random();
				trainlabel[i] = new int[]{1,0,0};
			}else if(train_N / classes <= i && i < (train_N / classes) * 2){
				traindata[i][0] = (float)g2.random();
				traindata[i][1] = (float)g1.random();
				trainlabel[i] = new int[]{0,1,0};
			}else{
				traindata[i][0] = (float)g3.random();
				traindata[i][1] = (float)g3.random();
				trainlabel[i] = new int[]{0,0,1};
			}
		}
		for(int i=0; i<test_N; i++){
			if(i < test_N / classes){
				//				for(int j=0; j < input_N; j++){
				//					traindata[i][j] = (float) g1.random();
				//				}
				testdata[i][0] = (float)g1.random();
				testdata[i][1] = (float)g2.random();
				testlabel[i] = new Integer[]{1,0,0};
			}else if(test_N / classes <= i && i < (test_N / classes) * 2){
				testdata[i][0] = (float)g2.random();
				testdata[i][1] = (float)g1.random();
				testlabel[i] = new Integer[]{0,1,0};
			}else{
				testdata[i][0] = (float)g3.random();
				testdata[i][1] = (float)g3.random();
				testlabel[i] = new Integer[]{0,0,1};
			}
		}

		//ミニバッチに分割
		for(int i=0; i< minibatch_N; i++){
			for(int j=0; j<minibatchsize; j++){
				train_minibatch[i][j] = traindata[minibatchindex.get(i*minibatchsize+j)];
				train_minibatch_label[i][j] = trainlabel[minibatchindex.get(i*minibatchsize+j)];
			}
		}
		Logistic_kaiki classifier = new Logistic_kaiki(input_N, output_N);
		//学習実行
		for(int epoch=0; epoch<epochs; epoch++){
			for(int batch=0; batch<minibatch_N; batch++){
				classifier.train(train_minibatch[batch], train_minibatch_label[batch], minibatchsize, l_rate);
			}
			l_rate = l_rate * 0.9f;
			System.out.println(epoch+"/"+epochs);
		}
		//テスト
		for(int i=0; i<test_N; i++){
			predict[i] = classifier.predict(testdata[i]);
		}
		print_result_test(predict,testlabel, test_N, classes);

	}


	public static void print_result_test(Integer[][] predict, Integer[][] testlabel, int test_N, int classes){
		int[][] confusion = new int[classes][classes];
		float accuracy =0.f;
		float[] precision = new float[classes];
		float[] recall = new float[classes];

		//モデルからの答えと正解を突き合わせ
		for(int i=0; i<test_N; i++){
			int row = Arrays.asList(predict[i]).indexOf(1);
			int column = Arrays.asList(testlabel[i]).indexOf(1);
			System.out.println(i +":"+ row+","+column);
			confusion[row][column]++;
		}

		for(int i=0; i<classes; i++){
			int col = 0, row = 0;

			for(int j=0; j<classes; j++){
				if(i==j){
					accuracy += confusion[i][j];
					precision[i] += confusion[j][i];
					recall[i] += confusion[i][j];
				}

				col += confusion[j][i];
				row += confusion[i][j];
			}

			precision[i] = precision[i]/(float)col;
			recall[i] = recall[i]/(float)row;
		}

		accuracy = accuracy / test_N;

		System.out.println("------------------------------------");
		System.out.println("Logistic Regression model evaluation");
		System.out.println("------------------------------------");
		System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
		System.out.println("Precision:");
		for (int i = 0; i < classes; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, precision[i] * 100);
		}
		System.out.println("Recall:");
		for (int i = 0; i < classes; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, recall[i] * 100);
		}

	}

	/**予測メソッド*/
	public Integer[] predict(float[] data) {
		// TODO 自動生成されたメソッド・スタブ

		float[] result = output(data);
		Integer[] label = new Integer[output_N];
		int index_max = -1;
		float max = 0.f;

		for(int i=0; i<output_N; i++){
			if(max < result[i]){
				max = result[i];
				index_max = i;
			}
			//System.out.println("result:"+ result[i]);
		}
		for(int i=0; i<output_N; i++){
			if(i == index_max){
				label[i] = 1;
			}else{
				label[i] = 0;
			}
			//System.out.println("label:"+ label[i]);
		}
		//System.out.println("predict end");
		return label;
	}

	/**予測メソッド*/
	public Integer[] predict(double[] data) {
		// TODO 自動生成されたメソッド・スタブ

		double[] result = output(data);
		Integer[] label = new Integer[output_N];
		int index_max = -1;
		double max = 0.;

		for(int i=0; i<output_N; i++){
			if(max < result[i]){
				max = result[i];
				index_max = i;
			}
			//System.out.println("result:"+ result[i]);
		}
		for(int i=0; i<output_N; i++){
			if(i == index_max){
				label[i] = 1;
			}else{
				label[i] = 0;
			}
			//System.out.println("label:"+ label[i]);
		}
		//System.out.println("predict end");
		return label;
	}

	/**
	 * 出力層のトレーニングメッソド。結果に基づきパラメータ更新
	 * Softmax-crossentropy
	 * @param data 入力データ
	 * @param label 正解ラベル
	 * @param minibatchsize ミニバッチサイズ
	 * @param l_rate 学習率
	 * @return 正解ラベルとの差異
	 */
	public float[][] train(float[][] data, int[][] label, int minibatchsize, float l_rate) {
		// TODO 自動生成されたメソッド・スタブ
		float[][] grad_weight = new float[output_N][input_N];
		float[] grad_bias = new float[output_N];
		float[][] error = new float[minibatchsize][output_N];

		for(int i=0; i<minibatchsize; i++){
			//出力層の数だけ出力を計算
			float[] result = output(data[i]);

			for(int j=0; j<output_N; j++){
				//それぞれの出力の誤差を計算
				error[i][j] = result[j] - label[i][j];

				for(int n=0; n<input_N; n++){
					//勾配を計算
					grad_weight[j][n] += error[i][j] * data[i][n];
				}
				//バイアスの勾配を計算
				grad_bias[j] += error[i][j];
			}
		}
		//パラメータの更新
		for(int i=0; i<output_N; i++){
			for(int j=0; j<input_N; j++){
				//weight[i][j] = weight[i][j] - l_rate * grad_weight[i][j] / minibatchsize;
				weight[i][j] -= l_rate * grad_weight[i][j] / minibatchsize;
			}
//			bias[i] = bias[i] - l_rate * grad_bias[i] / minibatchsize;
			bias[i] -= l_rate * grad_bias[i] / minibatchsize;
		}
		return error;
	}

	/**
	 * 出力層のトレーニングメッソド。結果に基づきパラメータ更新
	 * @param data 入力データ
	 * @param label 正解ラベル
	 * @param minibatchsize ミニバッチサイズ
	 * @param l_rate 学習率
	 * @return 正解ラベルとの差異
	 */
	public double[][] train(double[][] data, int[][] label, int minibatchsize, double l_rate) {
		// TODO 自動生成されたメソッド・スタブ
		double[][] grad_weight = new double[output_N][input_N];
		double[] grad_bias = new double[output_N];
		double[][] error = new double[minibatchsize][output_N];

		for(int i=0; i<minibatchsize; i++){
			//出力層の数だけ出力を計算
			double[] result = output(data[i]);

			for(int j=0; j<output_N; j++){
				//それぞれの出力の誤差を計算
				error[i][j] = result[j] - label[i][j];

				for(int n=0; n<input_N; n++){
					//勾配を計算
					grad_weight[j][n] += error[i][j] * data[i][n];
				}
				//バイアスの勾配を計算
				grad_bias[j] += error[i][j];
			}
		}
		//パラメータの更新
		for(int i=0; i<output_N; i++){
			for(int j=0; j<input_N; j++){
				weight[i][j] = (float) (weight[i][j] - l_rate * grad_weight[i][j] / minibatchsize);
			}
			bias[i] = (float) (bias[i] - l_rate * grad_bias[i] / minibatchsize);
		}
		return error;
	}

	/**
	 * 出力を計算
	 * @param input_data 入力データ
	 * @return 計算結果
	 */
	private float[] output(float[] input_data) {
		float[] activation = new float[output_N];

		for(int i=0; i< output_N; i++){
			for(int j=0; j< input_N; j++){
				//activation[i] = activation[i] + (weight[i][j] * input_data[j]);
				activation[i] += weight[i][j] * input_data[j];
				//System.out.println("weight:"+weight[i][j]);
			}
			activation[i] += bias[i];
			//System.out.println("act:"+activation[i]);
		}
		return ActivationFunction.softmax(activation, output_N);
	}

	/**
	 * 出力を計算
	 * @param input_data 入力データ
	 * @return 計算結果
	 */
	private double[] output(double[] input_data) {
		double[] activation = new double[output_N];

		for(int i=0; i< output_N; i++){
			for(int j=0; j< input_N; j++){
				activation[i] = activation[i] + (weight[i][j] * input_data[j]);
				//System.out.println("weight:"+weight[i][j]);
			}
			activation[i] += bias[i];
			//System.out.println("act:"+activation[i]);
		}
		return ActivationFunction.softmax(activation, output_N);
	}
}
