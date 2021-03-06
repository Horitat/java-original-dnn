package neuralnetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import single_layer.Logistic_kaiki;
import util.Common_method;
import Mersenne.Sfmt;

public class Multilayerperceptrons {

	int input_N;
	int hidden_N;
	int output_N;
	Sfmt mt;

	Hiddenlayer hiddenL;
	Logistic_kaiki outputL;



	public Multilayerperceptrons(int input, int hidden, int output, Sfmt m){
		input_N = input;
		hidden_N = hidden;
		output_N = output;

		if(m == null){
			int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
			m = new Sfmt(init_key);
		}
		mt = m;

		hiddenL = new Hiddenlayer(input, hidden, null, null, m, "tanh");
		outputL = new Logistic_kaiki(hidden, output);


	}


	public void train(float[][] X, int T[][], int minibatchSize, float l_rate){

		//中間層の出力 output of hidden layer
		float[][] Z = new float[minibatchSize][input_N];
		float[][] dy;

		//順伝播処理
		for(int i=0; i<minibatchSize; i++){
			Z[i] = hiddenL.forward(X[i]);
		}

		//出力を計算
		dy = outputL.train(Z, T, minibatchSize, l_rate);

		//逆伝播でパラメータを更新
		hiddenL.backward(X, Z, dy, outputL.weight, minibatchSize, l_rate);

	}

	public Integer[] predict(float[] x) {
		float[] z = hiddenL.output(x);
		return outputL.predict(z);
	}

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
		Sfmt mt = new Sfmt(init_key);

		//
		// Declare variables and constants
		//

		final int patterns = 2;
		final int train_N = 4;
		final int test_N = 4;
		final int nIn = 2;
		final int nHidden = 3;
		final int nOut = patterns;

		float[][] train_X;
		int[][] train_T;

		float[][] test_X;
		Integer[][] test_T;
		Integer[][] predicted_T = new Integer[test_N][nOut];

		final int epochs = 5000;
		float learningRate = 0.1f;

		final int minibatchSize = 1;  //  here, we do on-line training
		int minibatch_N = train_N / minibatchSize;

		float[][][] train_X_minibatch = new float[minibatch_N][minibatchSize][nIn];
		int[][][] train_T_minibatch = new int[minibatch_N][minibatchSize][nOut];
		List<Integer> minibatchIndex = new ArrayList<>();
		for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
		Common_method.list_shuffle(minibatchIndex, mt);

		//
		// Training simple XOR problem for demo
		//   class 1 : [0, 0], [1, 1]  ->  Negative [0, 1]
		//   class 2 : [0, 1], [1, 0]  ->  Positive [1, 0]
		//

		train_X = new float[][]{
				{0.f, 0.f},
				{0.f, 1.f},
				{1.f, 0.f},
				{1.f, 1.f}
		};
		train_T = new int[][]{
				{0, 1},
				{1, 0},
				{1, 0},
				{0, 1}
		};
		test_X = new float[][]{
				{0.f, 0.f},
				{0.f, 1.f},
				{1.f, 0.f},
				{1.f, 1.f}
		};
		test_T = new Integer[][]{
				{0, 1},
				{1, 0},
				{1, 0},
				{0, 1}
		};

		// create minibatches
		for (int i = 0; i < minibatch_N; i++) {
			for (int j = 0; j < minibatchSize; j++) {
				train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
				train_T_minibatch[i][j] = train_T[minibatchIndex.get(i * minibatchSize + j)];
			}
		}

		//
		// Build Multi-Layer Perceptrons model
		//

		// construct
		Multilayerperceptrons classifier = new Multilayerperceptrons(nIn, nHidden, nOut, mt);

		// train
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatch_N; batch++) {
				classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, learningRate);
			}
		}

		// test
		for (int i = 0; i < test_N; i++) {
			predicted_T[i] = classifier.predict(test_X[i]);
		}


		//
		// Evaluate the model
		//

		int[][] confusionMatrix = new int[patterns][patterns];
		double accuracy = 0.;
		double[] precision = new double[patterns];
		double[] recall = new double[patterns];

		for (int i = 0; i < test_N; i++) {
			int predicted_ = Arrays.asList(predicted_T[i]).indexOf(1);
			int actual_ = Arrays.asList(test_T[i]).indexOf(1);

			confusionMatrix[actual_][predicted_] += 1;
		}

		for (int i = 0; i < patterns; i++) {
			double col_ = 0.;
			double row_ = 0.;

			for (int j = 0; j < patterns; j++) {

				if (i == j) {
					accuracy += confusionMatrix[i][j];
					precision[i] += confusionMatrix[j][i];
					recall[i] += confusionMatrix[i][j];
				}

				col_ += confusionMatrix[j][i];
				row_ += confusionMatrix[i][j];
			}
			precision[i] /= col_;
			recall[i] /= row_;
		}

		accuracy /= test_N;

		System.out.println("--------------------");
		System.out.println("MLP model evaluation");
		System.out.println("--------------------");
		System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
		System.out.println("Precision:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, precision[i] * 100);
		}
		System.out.println("Recall:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, recall[i] * 100);
		}




	}
}
