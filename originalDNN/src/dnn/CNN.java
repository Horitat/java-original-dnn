package dnn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import neuralnetwork.Hiddenlayer;
import single_layer.Logistic_kaiki;
import util.Common_method;
import util.RandomGenerator;
import Mersenne.Sfmt;

public class CNN {

	int[] kernelnum;
	int[][] convkernelsize;
	int[][] poolkernelsize;
	int hidden_N;
	int output_N;

	/*
	 * 本来ならレイヤーのスーパークラスを作り、各レイヤーはそのクラスを継承
	 * スーパークラスの型を宣言し、ほしいレイヤーを格納する
	 */
	Convolutionlayer[] conv;
	MaxPoolinglayer[] pool;
	int[][] convoutsize;
	int[][] pooloutsize;
	int flatsize;
	Hiddenlayer hidden;
	Logistic_kaiki logistic;

	Sfmt mt;

	public CNN(int[] imageSize, int channel, int[] nKernels,
			int[][] kernelSizes, int[][] poolSizes, int nHidden, int nOut,
			Sfmt m, String activation) {
		// TODO 自動生成されたコンストラクター・スタブ

		if(m == null){
			int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
			m = new Sfmt(init_key);
		}
		mt = m;

		kernelnum = nKernels;
		convkernelsize = kernelSizes;
		poolkernelsize = poolSizes;
		hidden_N = nHidden;
		output_N = nOut;

		conv = new Convolutionlayer[kernelnum.length];
		pool = new MaxPoolinglayer[conv.length];
		convoutsize = new int[nKernels.length][imageSize.length];
		pooloutsize = new int[nKernels.length][imageSize.length];

		//畳込み層とプーリング層の初期化
		for(int i=0; i<nKernels.length; i++){
			int[] size;
			int chnl;

			if(i==0){
				size = new int[]{imageSize[0], imageSize[1]};
				chnl = channel;
			}else{
				size = new int[]{pooloutsize[i-1][0], pooloutsize[i-1][1]};
				chnl = nKernels[i-1];
			}

			//出力サイズ
			//outputsize = ((inputsize-kernelsize+2*paddingsize)/stridesize)+1
			convoutsize[i] = new int[]{size[0] - kernelSizes[i][0] + 1, size[1] - kernelSizes[i][1] + 1};
			pooloutsize[i] = new int[]{convoutsize[i][0] / poolSizes[i][0], convoutsize[i][1] / poolSizes[i][1]};
			System.out.println("convout:"+ (size[0] - kernelSizes[i][0] + 1)+","+ (size[1] - kernelSizes[i][1] + 1));
			System.out.println("poolout:"+ convoutsize[i][0] / poolSizes[i][0] +","+ convoutsize[i][1] / poolSizes[i][1]);

			conv[i] = new Convolutionlayer(size,chnl, nKernels[i], kernelSizes[i],poolSizes[i],convoutsize[i],pooloutsize[i], 1, mt, activation);
			pool[i] = new MaxPoolinglayer(poolkernelsize[i],pooloutsize[i], nKernels[i], 1, mt, "MAX", "");
		}

		//入力データを一次元に直すためのサイズ、全結合層への入力に使われる
		flatsize = nKernels[nKernels.length-1] * pooloutsize[pooloutsize.length-1][0]* pooloutsize[pooloutsize.length-1][1];
		System.out.println("flatsize:"+nKernels[nKernels.length-1] +":"+ pooloutsize[pooloutsize.length-1][0] +":"+ pooloutsize[pooloutsize.length-1][1]);

		hidden = new Hiddenlayer(flatsize, hidden_N, null, null, mt, activation);
		logistic = new Logistic_kaiki(hidden_N, output_N);
	}


	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
		Sfmt mt = new Sfmt(init_key);

		int train_N_each = 50;        // for demo
		int test_N_each = 10;          // for demo
		double pNoise_Training = 0.05;  // for demo
		double pNoise_Test = 0.10;     // for demo

		final int patterns = 3;

		final int train_N = train_N_each * patterns;
		final int test_N = test_N_each * patterns;

		final int[] imageSize = {32, 24};
		final int channel = 1;

		int[] nKernels = {10, 20};
		int[][] kernelSizes = { {3, 3}, {2, 2} };
		int[][] poolSizes = { {2, 2}, {2, 2} };

		int nHidden = 20;
		final int nOut = patterns;

		float[][][][] train_X = new float[train_N][channel][imageSize[0]][imageSize[1]];
		int[][] train_T = new int[train_N][nOut];

		float[][][][] test_X = new float[test_N][channel][imageSize[0]][imageSize[1]];
		Integer[][] test_T = new Integer[test_N][nOut];
		Integer[][] predicted_T = new Integer[test_N][nOut];


		int epochs = 500;
		float learningRate = 0.1f;

		final int minibatchSize = 25;
		int minibatch_N = train_N / minibatchSize;

		float[][][][][] train_X_minibatch = new float[minibatch_N][minibatchSize][channel][imageSize[0]][imageSize[1]];
		int[][][] train_T_minibatch = new int[minibatch_N][minibatchSize][nOut];
		List<Integer> minibatchIndex = new ArrayList<>();
		for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
		Common_method.list_shuffle(minibatchIndex, mt);


		//
		// Create training data and test data for demo.
		//
		float data=0;
		for (int pattern = 0; pattern < patterns; pattern++) {

			for (int n = 0; n < train_N_each; n++) {

				int n_ = pattern * train_N_each + n;

				for (int c = 0; c < channel; c++) {

					for (int i = 0; i < imageSize[0]; i++) {

						for (int j = 0; j < imageSize[1]; j++) {

							if ((i < (pattern + 1) * (imageSize[0] / patterns)) && (i >= pattern * imageSize[0] / patterns)) {
								train_X[n_][c][i][j] = (float) (((int) 128. * mt.NextUnif() + 128.) * RandomGenerator.binomial(1, 1 - pNoise_Training, mt) / 256.);
							} else {
								train_X[n_][c][i][j] = (float) (128. * RandomGenerator.binomial(1, pNoise_Training, mt) / 256.);
							}
							System.out.println(data);
							data += 1.f;
						}
					}
				}

				for (int i = 0; i < nOut; i++) {
					if (i == pattern) {
						train_T[n_][i] = 1;
					} else {
						train_T[n_][i] = 0;
					}
				}
			}
			data = 0.f;
			for (int n = 0; n < test_N_each; n++) {

				int n_ = pattern * test_N_each + n;

				for (int c = 0; c < channel; c++) {

					for (int i = 0; i < imageSize[0]; i++) {

						for (int j = 0; j < imageSize[1]; j++) {

							if ((i < (pattern + 1) * imageSize[0] / patterns) && (i >= pattern * imageSize[0] / patterns)) {
								test_X[n_][c][i][j] = (float) (((int) 128. * mt.NextUnif() + 128.) * RandomGenerator.binomial(1, 1 - pNoise_Test, mt) / 256.);
							} else {
								test_X[n_][c][i][j] = (float) (128. * RandomGenerator.binomial(1, pNoise_Test, mt) / 256.);
							}
							data += 1.f;
						}
					}
				}

				for (int i = 0; i < nOut; i++) {
					if (i == pattern) {
						test_T[n_][i] = 1;
					} else {
						test_T[n_][i] = 0;
					}
				}
			}
		}


		// create minibatches
		for (int j = 0; j < minibatchSize; j++) {
			for (int i = 0; i < minibatch_N; i++) {
				train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
				train_T_minibatch[i][j] = train_T[minibatchIndex.get(i * minibatchSize + j)];
			}
		}


		//
		// Build Convolutional Neural Networks model
		//

		// construct CNN
		System.out.println("Building the model...");
		CNN classifier = new CNN(imageSize, channel, nKernels, kernelSizes, poolSizes, nHidden, nOut, mt, "ReLU");
		System.out.println("done.");

		//System.exit(-1);
		// train the model
		System.out.println("Training the model...");
		System.out.println();

		for (int epoch = 0; epoch < epochs; epoch++) {

			if ((epoch + 1) % 50 == 0) {
				System.out.println("\titer = " + (epoch + 1) + " / " + epochs);
			}

			for (int batch = 0; batch < minibatch_N; batch++) {
				classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, learningRate);
			}
			learningRate *= 0.999;
		}
		System.out.println("done.");

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
		System.out.println("CNN model evaluation");
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



	/**
	 *
	 * @param x [minibatchsize][chanel][imgsize][imgsize]
	 * @param label
	 * @param minibatchsize
	 * @param l_rate
	 */
	public void train(float[][][][] x, int[][] label, int minibatchsize, float l_rate){
		//各層のインプットデータ,活性化後をキャッシュ
		List<float[][][][]> preactdata = new ArrayList<>(kernelnum.length);
		List<float[][][][]> after_act = new ArrayList<>(kernelnum.length);
		//入力データを保持のための+1
		List<float[][][][]> downsampling = new ArrayList<>(kernelnum.length+1);
		downsampling.add(x);

		//初期化
		for(int i=0; i<kernelnum.length; i++){
			preactdata.add(new float[minibatchsize][kernelnum[i]][convoutsize[i][0]][convoutsize[i][1]]);
			after_act.add(new float[minibatchsize][kernelnum[i]][convoutsize[i][0]][convoutsize[i][1]]);
			downsampling.add(new float[minibatchsize][kernelnum[i]][convoutsize[i][0]][convoutsize[i][1]]);
		}

		//一次元に変換したデータをキャッシュ用
		float[][] flatdata = new float[minibatchsize][flatsize];
		//隠れ層の出力キャッシュ用
		float[][] hiddendata = new float[minibatchsize][hidden_N];
		//出力層の逆伝播
		float[][] dy;
		//隠れ層の逆伝播
		float[][] dz;
		//一次元変換の逆伝播
		float[][] dx_flat = new float[minibatchsize][flatsize];
		//畳込み、プーリングの逆伝播
		float[][][][] dx = new float[minibatchsize][kernelnum[kernelnum.length-1]][pooloutsize[pooloutsize.length-1][0]][pooloutsize[pooloutsize.length-1][1]];

		float[][][][] dc;

		//batchnormがありかなしかで条件分けし、ミニバッチでループか各層をミニバッチ単位でループするか
		for(int n=0; n<minibatchsize; n++){
		//順伝播
			float[][][] z = x[n].clone();
			for(int i=0; i<kernelnum.length; i++){
				z = conv[i].forward(z, preactdata.get(i)[n], after_act.get(i)[n]);
				z = pool[i].maxpooling(z);

				downsampling.get(i+1)[n] = z.clone();
			}

			float[] xx = data_flat(z);
			flatdata[n] = xx.clone();
			hiddendata[n] = hidden.forward(xx);
		}
		//出力層の順伝播、逆伝播
		dy = logistic.train(hiddendata, label, minibatchsize, l_rate);
		//全結合層の逆伝播
		dz = hidden.backward(flatdata, hiddendata, dy, logistic.weight, minibatchsize, l_rate);

		//畳込み層、プーリング層の逆伝播のために、フラット化したデータを戻す
		for(int n=0; n<minibatchsize; n++){
			for(int i=0; i<flatsize; i++)
				for(int j=0; j<hidden_N; j++){
					dx_flat[n][i] += hidden.weight[j][i] * dz[n][j];
				}
			dx[n] = data_unflat(dx_flat[n]);
		}

		//畳込み層の逆伝播
		dc = dx.clone();
		for(int i= kernelnum.length-1; i>-1; i--){
			float[][][][] poolback = pool[i].backmaxpooing( after_act.get(i), downsampling.get(i+1), dc, convoutsize[i], minibatchsize);
			dc = conv[i].backward(downsampling.get(i), preactdata.get(i), after_act.get(i), downsampling.get(i+1), poolback,
					minibatchsize, l_rate);
		}
	}


	private Integer[] predict(float[][][] x) {
		// TODO 自動生成されたメソッド・スタブ
		List<float[][][]> preact = new ArrayList<>(kernelnum.length);
		List<float[][][]> act = new ArrayList<>(kernelnum.length);

		for(int i=0; i<kernelnum.length; i++){
			preact.add(new float[kernelnum[i]][convoutsize[i][0]][convoutsize[i][1]]);
			act.add(new float[kernelnum[i]][convoutsize[i][0]][convoutsize[i][1]]);
		}

		float[][][] z = x.clone();

		for(int i=0; i<kernelnum.length; i++){
			z = conv[i].forward(z, preact.get(i), act.get(i));
		}
		return logistic.predict(hidden.forward(data_flat(z)));
	}


	private float[] data_flat(float[][][] z) {
		// TODO 自動生成されたメソッド・スタブ
		float[] f = new float[flatsize];
		//nKernels[nKernels.length-1] +":"+ pooloutsize[pooloutsize.length-1][0] +":"+ pooloutsize[pooloutsize.length-1][1]
		//System.out.println(z.length+":"+z[0].length+":"+z[0][0].length);
		int n=0;
		for(int k=0; k<kernelnum[kernelnum.length-1]; k++)
			for(int i=0; i<pooloutsize[pooloutsize.length-1][0]; i++)
				for(int j=0; j<pooloutsize[pooloutsize.length-1][1]; j++){
					f[n] = z[k][i][j];
					n++;
				}

		return f;
	}


	public float[][][] data_unflat(float[] x){
		float[][][] z = new float[kernelnum[kernelnum.length-1]][pooloutsize[pooloutsize.length-1][0]][pooloutsize[pooloutsize.length-1][1]];
		int n=0;

		for(int k=0; k<z.length; k++)
			for(int i=0; i<z[0].length; i++)
				for(int j=0; j<z[0][0].length; j++){
					z[k][i][j] = x[n];
					n++;
				}

		return z;
	}

}
