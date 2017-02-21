package layers;

public class BatchNormlayer {

	private float gannma;
	private float beta;
	private float[][][][] xc,xn;
	private float std;
	final private float param = (float) 1e-8;;

	public BatchNormlayer(){
		gannma = 1;
		beta = 0;

	}

	/**
	 * バッチノーマライゼーション
	 * @param z 入力値
	 * @param minibatch バッチ数
	 * @param chanel チャネル数
	 * @return 結果
	 */
	public float[][][][] forward(float[][][][] z, int minibatch, int chanel){
		int size = z[0][0].length, size1 = z[0][0][0].length;
		float mu = 0;
		for(int mb_size =0; mb_size<minibatch; mb_size++)
			for(int c=0; c<chanel; c++)
				for(int i=0; i<size; i++)
					for(int j=0; j<size1; j++){
						mu += (float)(z[mb_size][c][i][j] / minibatch);
					}

		float delta = 0;
		for(int mb_size =0; mb_size<minibatch; mb_size++)
			for(int c=0; c<z[mb_size].length; c++)
				for(int i=0; i<size; i++)
					for(int j=0; j<size1; j++){
						xc[mb_size][c][i][j] = (float)((z[mb_size][c][i][j] - mu));
						delta += (float)(xc[mb_size][c][i][j]* xc[mb_size][c][i][j] / minibatch);
					}

		std = (float) Math.sqrt(delta -param);
		float[][][][] result = new  float[minibatch][chanel][size][size1];

		for(int mb_size =0; mb_size<minibatch; mb_size++)
			for(int c=0; c<z[mb_size].length; c++)
				for(int i=0; i<size; i++)
					for(int j=0; j<size1; j++){
						result[mb_size][c][i][j] = xc[mb_size][c][i][j] / std;
					}

		return result;
	}
}
