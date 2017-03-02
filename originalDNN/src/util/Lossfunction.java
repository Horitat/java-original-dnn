package util;

public class Lossfunction {

	public interface FloatFunction<R, S> {

		/**
		 * Applies this function to the given argument.
		 *
		 * @param value the function argument
		 * @return the function result
		 */
		R apply(float value, float label);
	}


	public static class MSE{
		public static float mse(float data, float label){
			return (label - data)*(label - data)/2.f;
		}

		public static float dmse(float data, float label){
			return data-label;
		}
	}

	public static class Cross_Entropy{

		public static float cross_entropy_multi(float data, float label){
			return (float) (-label * Math.log(data));
		}

		public static float dcross_entropy_multi(float data, float label){
			return (- label)/ data;
		}

		public static float cross_entropy(float data, float label){
			return (-1.f)*(float)(label * Math.log(data)+(1-label)*Math.log(1-data));
		}

		public static float dcross_entropy(float data, float label){
			return (data - label)/ (data * (1-data));
		}
	}
	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ

	}

}
