package single_layer;

public class Loss_Function {


	public static class MSE{
		public static float mse(float data, float label){
			return (data - label)*(data - label)/2.f;
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
}
