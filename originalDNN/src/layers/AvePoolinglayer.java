package layers;

import org.apache.commons.lang3.StringUtils;

import util.ActivationFunction;
import Mersenne.Sfmt;

public class AvePoolinglayer {
	int[] poolkernelsize;
	int[] pooloutsize;
	Sfmt mt;
	int flatsize;
	int kernelnum;
	String type;
	FloatFunction<Float> activation;
	FloatFunction<Float> dactivation;
	int stride;
	int avesize;

	@FunctionalInterface
	public interface FloatFunction<R> {

		/**
		 * Applies this function to the given argument.
		 *
		 * @param value the function argument
		 * @return the function result
		 */
		R apply(float value);
	}

	/**
	 * プーリング層のコンストラクタ
	 * @param pkernelsize プーリング層のカーネルサイズ
	 * @param poutsize プーリング層の出力数
	 * @param m メルセンヌツイスタ
	 */
	public AvePoolinglayer(int[] pkernelsize, int[] poutsize, int nkernel, int stride, Sfmt m, String poolingtype, String actfunc){
		poolkernelsize = pkernelsize;
		pooloutsize = poutsize;
		kernelnum = nkernel;
		mt = m;
		avesize = pkernelsize[0]*pkernelsize[1];

		if(stride <= 0){
			this.stride = 1;
		}else{
			this.stride = stride;
		}

		if(actfunc.equals("sigmoid")){
			activation = (float x)->ActivationFunction.logistic_sigmoid(x);
			dactivation = (float x)->ActivationFunction.dsigmoid(x);
		}else if(actfunc.equals("tanh")){
			activation = (float x)->ActivationFunction.tanh(x);
			dactivation = (float x)->ActivationFunction.dtanh(x);
		}else if(actfunc.equals("ReLU")){
			activation = (float x)->ActivationFunction.ReLU(x);
			dactivation = (float x)->ActivationFunction.dReLU(x);
		}else{
			activation = null;
			dactivation = null;
		}

		if(poolingtype.equals("MAX") || StringUtils.isEmpty(poolingtype)){
			System.out.println("MAX");
			type = "MAX";
		}else if(poolingtype.equals("AVE")){
			type = "AVE";
		}else{
			System.out.println("error");
			throw new IllegalArgumentException("specify poolingtype function");
		}

	}

	/**
	 * マックスプーリングで順伝播
	 * @param z 入力値
	 * @return 順伝播の値
	 */
	public float[][][] avepooling(float[][][] z) {
		// TODO 自動生成されたメソッド・スタブ
		float[][][] y = new float[kernelnum][pooloutsize[0]][pooloutsize[1]];

		for(int kernel=0; kernel<kernelnum; kernel++)
			for(int i=0; i<pooloutsize[0]; i=i+stride)
				for(int j=0; j<pooloutsize[1]; j=j+stride){
//					for(int i=0; i<pooloutsize[0]; i++)
//						for(int j=0; j<pooloutsize[1]; j++){
					float ave = 0.f;

					for(int ks0=0; ks0<poolkernelsize[0]; ks0++)
						for(int ks1=0; ks1<poolkernelsize[1]; ks1++){
//							if(ks0==0 && ks1==0){
//								ave = z[kernel][poolkernelsize[0]*i][poolkernelsize[1]*j];
//								continue;
//							}

//							if(ave < z[kernel][poolkernelsize[0]*i+ks0][poolkernelsize[1]*j+ks1]){
								ave += z[kernel][poolkernelsize[0]*i+ks0][poolkernelsize[1]*j+ks1];
//							}
						}

					y[kernel][i][j] = ave / avesize;
				}
		return y;
	}

	/**
	 * アベレージプーリングの逆伝播
	 * @param x 入力
	 * @param y 出力
	 * @param dy 逆伝播の値
	 * @param convoutsize 畳込み層のアウトプット数
	 * @param minibatchsize ミニバッチサイズ
	 * @return 逆伝播の値
	 */
	public float[][][][] backmaxpooing(float[][][][] x, float[][][][]y, float[][][][] dy,int[] convoutsize, int minibatchsize){
		float[][][][] back = new float[minibatchsize][kernelnum][convoutsize[0]][convoutsize[1]];

		for(int n=0; n<minibatchsize; n++)
			for(int kernel=0; kernel<kernelnum; kernel++)
				for(int i=0; i<pooloutsize[0]; i++)
					for(int j=0; j<pooloutsize[1]; j++)

						for(int s=0; s<poolkernelsize[0]; s++)
							for(int t=0; t<poolkernelsize[1]; t++){
								float d = 0.f;
								//System.out.println(n+":"+kernel+":"+(poolkernelsize[0]*i+s)+":"+(poolkernelsize[1]*j+t));
								if(y[n][kernel][i][j] == x[n][kernel][poolkernelsize[0]*i+s][poolkernelsize[1]*j+t]){
									d = dy[n][kernel][i][j];
								}
								back[n][kernel][poolkernelsize[0]*i+s][poolkernelsize[1]*j+t] = d;
							}

		return back;
	}


}
