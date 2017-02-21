package dnn;

import neuralnetwork.Hiddenlayer;

import org.apache.commons.lang3.StringUtils;

import util.ActivationFunction;
import util.RandomGenerator;
import Mersenne.Sfmt;

//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.factory.Nd4j;


public class Convolutionlayer {

	int[] imagesize;
	float[][][][] weight;
	float[][][][] adagrad_gradient;
	float[] bias;
	int chanel;
	int[] kernelsize;
	int[] convoutsize;
	int kernelnum;
	Sfmt mt;
	Hiddenlayer.FloatFunction<Float> activation;
	Hiddenlayer.FloatFunction<Float> dactivation;
	int flatsize;
	int stride;

	/**
	 * 畳込み層のコンストラクタ
	 * @param imgsize 入力画像サイズ
	 * @param chnl チャネル数
	 * @param nkernel カーネルの数
	 * @param kernelsize 畳込み層のカーネルサイズ
	 * @param pooloutsize プーリング層の出力数
	 * @param convoutsize 畳込み層の出力数
	 * @param poolkernelsize プーリング層のカーネルサイズ
	 * @param m メルセンヌツイスタ
	 * @param actfunc 活性化関数
	 */
	public Convolutionlayer(int[] imgsize, int chnl, int nkernel, int[] kernelsize, int[] pooloutsize, int[] convoutsize,
			int[] poolkernelsize, int stride, Sfmt m, String actfunc) {
		// TODO 自動生成されたコンストラクター・スタブ
		if(m == null){
			int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
			m = new Sfmt(init_key);
		}
		mt = m;

		if(bias == null){
			bias = new float[nkernel];
		}

		kernelnum = nkernel;
		chanel = chnl;
		this.kernelsize = kernelsize;
		imagesize = imgsize;
		this.convoutsize = convoutsize;

		if(stride <= 0){
			this.stride = 1;
		}else{
			this.stride = stride;
		}

		if(weight == null){
			weight = new float[nkernel][chnl][kernelsize[0]][kernelsize[1]];
			adagrad_gradient = new float[nkernel][chnl][kernelsize[0]][kernelsize[1]];
			float in  = (float)(chnl * kernelsize[0] * kernelsize[1]);
			float out = (float)(nkernel * kernelsize[0] * kernelsize[1] / (convoutsize[0] * convoutsize[1]));
			float w = (float) Math.sqrt(6./(in+out));

			for(int kernel=0; kernel<nkernel; kernel++)
				for(int c=0; c<chnl; c++)
					for(int ksize0=0; ksize0<kernelsize[0]; ksize0++)
						for(int ksize1=0; ksize1<kernelsize[1]; ksize1++){
							weight[kernel][c][ksize0][ksize1]=RandomGenerator.uniform(-w, w, m);
							adagrad_gradient[kernel][c][ksize0][ksize1] = 0.f;
						}
		}

		System.out.println("conv constractor" + convoutsize[0] +":"+ convoutsize[1]);

		if(actfunc.equals("sigmoid")){
			activation = (float x)->ActivationFunction.logistic_sigmoid(x);
			dactivation = (float x)->ActivationFunction.dsigmoid(x);
		}else if(actfunc.equals("tanh")){
			activation = (float x)->ActivationFunction.tanh(x);
			dactivation = (float x)->ActivationFunction.dtanh(x);
		}else if(actfunc.equals("ReLU")){
			activation = (float x)->ActivationFunction.ReLU(x);
			dactivation = (float x)->ActivationFunction.dReLU(x);
		}else if(StringUtils.isEmpty(actfunc)){
			throw new IllegalArgumentException("specify activation function");
		}else{
			throw new IllegalArgumentException("activation function not supported");
		}
	}

	public float[][][] forward(float[][][] z, float[][][] preactdata, float[][][] after_act) {
		// TODO 自動生成されたメソッド・スタブ
		return convolve(z, preactdata, after_act);
	}

	public float[][][] convolve(float[][][] z, float[][][] preactdata, float[][][] after_act){

		float[][][] y = new float[kernelnum][convoutsize[0]][convoutsize[1]];

		/*
		 * ストライドを加えるなら、Zのi,j部分にストライドを加える
		 * 配列オーバー部分には0を加える
		 */
		for(int kernel=0; kernel<kernelnum; kernel++){
			//怪しい.convoutではなくimgサイズ？zの長さではないか？
			for(int i=0; i<convoutsize[0]; i=i+stride)
//				for(int i=0; i<convoutsize[0]; i++)
				for(int j=0; j<convoutsize[1]; j=j+stride){
//					for(int j=0; j<convoutsize[1]; j++){
					float convol = 0.f;

					for(int c=0; c<chanel; c++){
						for(int ks0=0; ks0<kernelsize[0]; ks0++)
							for(int ks1=0; ks1<kernelsize[1]; ks1++){
								//System.out.println(kernel+":"+(i+ks0)+":"+(j+ks1));
								convol += weight[kernel][c][ks0][ks1] * z[c][i+ks0][j+ks1];
							}
					}
					//活性化前後でキャッシュ
					//System.out.println(kernel+":"+i+":"+j);
					preactdata[kernel][i][j] = convol + bias[kernel];
					after_act[kernel][i][j] = activation.apply(preactdata[kernel][i][j]);
					y[kernel][i][j] = after_act[kernel][i][j];
				}
		}
		return y;
	}

	/**
	 *畳込み層の逆伝播を行う
	 * @param x 入力値
	 * @param preact 活性化する前の出力
	 * @param after_act 活性化後の出力
	 * @param downsampling マックスプーリングの値
	 * @param dy 逆伝播の値
	 * @param minibatchsize
	 * @param l_rate
	 * @return
	 */
	public float[][][][] backward(float[][][][] x , float[][][][] preact, float[][][][] after_act,
			float[][][][] downsampling, float[][][][] dy, int minibatchsize, float l_rate) {
		// TODO 自動生成されたメソッド・スタブ
		return deconvolve(x, preact, dy, minibatchsize, l_rate);
	}


	private float[][][][] deconvolve(float[][][][] x, float[][][][] preact, float[][][][] dy,
			int minibatchsize, float l_rate){
		float[][][][] grad_weight = new float[kernelnum][chanel][kernelsize[0]][kernelsize[1]];
		float[] grad_bias = new float[kernelnum];

		//重みとバイアスの勾配を計算.ここでアップデータの分岐を行うか
		//分岐をクラス生成時に選択するようにする
		for(int n=0; n<minibatchsize; n++){
			for(int k=0; k<kernelnum; k++)
				for(int i=0; i<convoutsize[0]; i++)
					for(int j=0; j<convoutsize[1]; j++){
						//活性化の微分
//						System.out.println(n+":"+k+":"+i+":"+j);
						float d_ = dy[n][k][i][j] * dactivation.apply(preact[n][k][i][j]);
						grad_bias[k] += d_; //バイアスの勾配
						//重みの勾配
						for(int c=0; c<chanel; c++)
							for(int s=0; s<kernelsize[0]; s++)
								for(int t=0; t<kernelsize[1]; t++){
									grad_weight[k][c][s][t] += d_ * x[n][c][i+s][j+s];
								}
					}
		}

		//パラメータの更新
		for(int k=0; k<kernelnum; k++){
			bias[k] -= l_rate * grad_bias[k] / minibatchsize;

			for(int c=0; c<chanel; c++)
				for(int s=0; s<kernelsize[0]; s++)
					for(int t=0; t<kernelsize[1]; t++){
						weight[k][c][s][t] -= l_rate * grad_weight[k][c][s][t] / minibatchsize;
					}
		}

		float [][][][] dx = new float[minibatchsize][chanel][imagesize[0]][imagesize[1]];
		//誤差を計算
		for(int n=0; n<minibatchsize; n++)
			for(int c=0; c<chanel; c++)
				for(int i=0; i<imagesize[0]; i++)
					for(int j=0; j<imagesize[1]; j++){

						for(int k=0; k<kernelnum; k++)
							for(int s=0; s<kernelsize[0]; s++)
								for(int t=0; t<kernelsize[1]; t++){
									float d_ = 0;

									if(i-(kernelsize[0]-1)-s>=0 && j-(kernelsize[1]-1)-t>=0){
										d_ = dy[n][k][i-(kernelsize[0]-1)-s][j-(kernelsize[1]-1)-t]
												* dactivation.apply(preact[n][k][i-(kernelsize[0]-1)-s][j-(kernelsize[1]-1)-t])
												* weight[k][c][s][t];
									}
									dx[n][c][i][j] += d_;
								}
					}


		return dx;
	}
}
