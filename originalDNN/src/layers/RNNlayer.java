package layers;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.StringUtils;

import util.ActivationFunction;
import util.Lossfunction;
import util.RandomGenerator;
import Mersenne.Sfmt;

public class RNNlayer {
	int input_N;
	int hidden_N;
	int output_N;
	int selfloop_output;

	public float[][] weight;
	public float[][] reccurent_weight;
	public float[] bias;
	public float[] recurrent_bias;
	Sfmt mt;
	public FloatFunction<Float> activation;
	public FloatFunction<Float> dactivation;
	public Lossfunction.FloatFunction<Float,Float> dlossfunc;

	//次に入力されるループ用
	private float[] loop_input;

	List<float[][]> bptt_compute = new ArrayList<>();
	List<float[]> past_input = new ArrayList<>();
	//時刻t
	private int t;
	private int ago = 3;

	/*
	 * リカレントの誤差計算のためのラベル
	 * 出力によって変更できるように設計するべきかどうか
	 */
//	train_T_minibatch = new int[minibatch_N][minibatchSize][nOut];
	float[][] label;



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


	public RNNlayer(int input, int output, float[][] W, float[][] r_w, float[] b, String actfunc, Sfmt m){
		if(m == null){
			int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
			m = new Sfmt(init_key);
		}
		//mt = m;

		//重みの初期化
		if(W == null){
			W = new float[output][input];
			float min = 1.f/ input;

			for(int i=0; i<output; i++){
				for(int j=0; j<input; j++){
					W[i][j] = RandomGenerator.uniform(-min, min, m);
				}
			}
		}
		if(r_w == null){
			r_w = new float[input][input];
			float min = 1.f/ input;

			for(int i=0; i<input; i++){
				for(int j=0; j<input; j++){
					r_w[i][j] = RandomGenerator.uniform(-min, min, m);
				}
			}
		}

		//バイアスの初期化
		if(b == null){
			b = new float[output];
		}

		input_N= input;
		output_N=output;

		bias=b;
		mt=m;
		selfloop_output = input;
		t = 0;

		weight=W;
		reccurent_weight = r_w;


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

	/**
	 * 順伝播
	 * @param x 入力データ
	 * @return 出力
	 */
	public float[] forward(float[] x) {
		// TODO 自動生成されたメソッド・スタブ
		return output(x);
	}

	public float[] output(float[] x) {
		// TODO 自動生成されたメソッド・スタブ
		float y[] = new float[output_N];
		//loop_inputを活性化した入力値
		float[] activate_loop = new float[input_N];
		if(t != 0){
			//ループの入力に重みをかける
			for(int i=0; i<input_N; i++)
				for(int j=0; j<input_N; j++){
					activate_loop[j] = loop_input[j] * reccurent_weight[i][j];
				}
			//biasを足したい場合はここで+recurrent_bias
		}

		for(int i=0; i<output_N; i++){
			float act = 0.f;
			//y[i] = 0.f;
			for(int j=0; j<input_N; j++){
				//y[i] += weight[i][j] * x[j];
				if(t == 0){
					act += weight[i][j] * x[j];
					activate_loop[j] = act;
				}else{
					act += weight[i][j] * x[j] + activate_loop[j];
					activate_loop[j] = act;
				}
			}
			//y[i] += bias[i];
			act += bias[i];
			y[i] = activation.apply(act);
		}
		//誤差逆伝播に使用
		if(past_input.size() >= ago+1){
			past_input.remove(0);
		}
		past_input.add(activate_loop);
		//次のリカレントの入力に使用
		loop_input = activate_loop.clone();
		t++;
		System.out.println(loop_input.length);
		return y;
	}

	public int[] outputBinomial(int[] x, Sfmt mt) {

		int[] y = new int[output_N];

		float[] xCast = new float[x.length];
		for (int i = 0; i < xCast.length; i++) {
			xCast[i] = (float) x[i];
		}

		float[] out = output(xCast);

		for (int j = 0; j < output_N; j++) {
			y[j] = RandomGenerator.binomial(1, out[j], mt);
		}

		return y;
	}

	/**
	 * 逆伝播メソッド
	 * 順伝播の重みを更新する。リカレントの重みはここではしない
	 * @param x この層への入力
	 * @param z この層の出力
	 * @param dy 前の層（出力に近い層）の逆伝播の値(誤差)
	 * @param minibatchSize ミニバッチサイズ
	 * @param l_rate 学習率
	 * @return 逆伝播の結果
	 */
	public float[][] backward(float[][] x, float[][] z, float[][] dy, int minibatchSize, float l_rate) {
		// TODO 自動生成されたメソッド・スタブ
		//逆伝播する誤差
//		float[][] dz = new float[minibatchSize][output_N];
		//さかのぼる時刻の設定
		//歴代誤差を記憶
		if(bptt_compute.size() >= ago){
			bptt_compute.remove(0);
		}
		bptt_compute.add(dy);

		return BPTT(minibatchSize, dy, l_rate);
//		float[][] grad_w = new float[output_N][input_N];
//		float[] grad_b = new float[output_N];

		/*
		//SGD
		for(int n=0; n<minibatchSize; n++){
			for(int i=0; i<output_N; i++){
				//前の層のアウトプット数
				for(int p=0; p<dy[0].length; p++){
					//前の層からの逆伝播
					dz[n][i] += weight_prev[p][i] * dy[n][p];
				}
				//出力を微分
				dz[n][i] *= dactivation.apply(z[n][i]);

				for(int j=0; j<input_N; j++){
					if(t==0){
						grad_w[i][j] += dz[n][i] * x[n][j];
					}else{
						grad_w[i][j] += dz[n][i] * (x[n][j] + activate_loop[j]);
					}
				}
				grad_b[i] += dz[n][i];
			}
		}*/
//		return dz;
	}

	/**
	 * Real Time Recurrent Learning
	 * @param minibatch
	 * @return
	 */
	public float[][] RTRL(int minibatch, float[][] x, float[][] dy, float l_rate){
		//逆伝播する誤差
		float[][] dz = new float[minibatch][output_N];




		return dz;
	}

	/**
	 * BackPropagation Through Time
	 * @param minibatch ミニバッチサイズ
	 * @param dy 前の層の誤差
	 * @param l_rate 学習率
	 * @return 順伝播用の誤差
	 */
	public float[][] BPTT(int minibatch, float[][] dy, float l_rate){
		int step = bptt_compute.size();
		//逆伝播する誤差
		float[][] dz = new float[minibatch][output_N];
		float[][][] dz_step = new float[step][minibatch][input_N];

		float[][][] past_dz = new float[step][minibatch][input_N];

		//次の層からの逆伝播
		float[][][] past_dy = new float[step][minibatch][input_N];

		//式7.7の第一項
		//月曜日に第2項以降を作る
		for(int s=0; s<step; s++){
			for(int m=0; m<minibatch; m++)
				for(int i=0; i< input_N; i++)
					for(int p=0; p<dy[0].length; p++){
//						past_dy[s][m][i] += weight_prev[p][i] * bptt_compute.get(step)[m][p];//dy[m][p];

						past_dy[s][m][i] += weight[p][i] * bptt_compute.get(s)[m][p];
						dz[m][p] += weight[p][i] * bptt_compute.get(s)[m][p];
					}
		}

		//第2項はこのリカレント層のデルタは、出力と正解ラベルの誤差関数の微分
		//つまり問題によって誤差関数が変わる。
		//https://micin.jp/feed/developer/articles/rnn000 の22式より
		for(int s=step-1; s>=0; s--){
			for(int m=0; m<minibatch; m++){
				for(int i=0; i< input_N; i++){
					System.out.println(s+":"+m+":"+i+":");
					if(s == ago-1 || step == 1){
						past_dz[s][m][i] = 0;
					}else{
						past_dz[s][m][i] = dz_step[s+1][m][i] * reccurent_weight[i][i] * dactivation.apply(past_input.get(s)[i]);
					}
					//デルタを計算
					dz_step[s][m][i] = past_dy[s][m][i] + past_dz[s][m][i];
				}

			}
		}

		float[][] grad_weight = new float[input_N][input_N];
		for(int s=0; s<step; s++)
			for(int m=0; m<minibatch; m++)
				for(int i=0; i<input_N; i++)
					for(int j=0; j<input_N; j++){
						grad_weight[i][j] += dz_step[s][m][i] * past_input.get(s)[i];
					}

		//リカレントの重みを更新
		for(int i=0; i<input_N; i++)
			for(int j=0; j<input_N; j++){
				reccurent_weight[i][j] -= l_rate * grad_weight[i][j] / minibatch;
			}

		//順伝播重み誤差
		float[][] grad_w = new float[output_N][input_N];
		float[] grad_b = new float[output_N];//バイアス
		for(int s=0; s<step; s++)
			for(int m=0; m<minibatch; m++)
				for(int i=0; i<output_N; i++)
					for(int j=0; j<input_N; j++){
						grad_w[i][j] += dz_step[s][m][j] * past_input.get(s+1)[j];
						grad_b[i] += dz_step[s][m][j];
						dz[m][i] +=  dz_step[s][m][j];
					}
		//バイアスの更新
		//重みとバイアスを更新
		for(int i=0; i<output_N; i++){
			for(int j=0; j<input_N; j++){
				weight[i][j] -= l_rate * grad_w[i][j] / minibatch;
			}
			bias[i] -= l_rate * grad_b[i] / minibatch;
		}

		//一番最新の誤差を返す
		return dz;
	}

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		//RNNlayer(int input, int output, float[][] W, float[][] r_w, float[] b, String actfunc, Sfmt m)
		int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
		Sfmt mt = new Sfmt(init_key);
		RNNlayer rnn = new RNNlayer(5, 3, null, null, null, "ReLU", mt);

		float[][] x = new float[2][rnn.input_N];
		float[][] z = new float[2][rnn.output_N];
		float[][] dy = new float[2][rnn.output_N];

		for(int m=0; m<2; m++){
			for(int i=0; i<rnn.input_N; i++){
				x[m][i] = (float)mt.NextUnif();
			}
		}

		for(int epoch =0; epoch < 10; epoch++)
			for(int m=0; m<2; m++){
				z[m] = rnn.forward(x[m]);
			}

		float[][] error = rnn.backward(x, z, dy, 2, 0.01f);


		System.out.printf("error[%d][%d]\n", error.length, error[0].length );
		for(int i=0; i<error.length; i++)
			for(int j=0; j<error[0].length; j++){
				System.out.println("["+i+"]["+j+"]:"+error[i][j]);
			}


		System.out.println("\nrnn test end");
	}

}
