package layers;

import org.apache.commons.lang3.StringUtils;

import updater.Updater;
import util.ActivationFunction;
import util.Common_method;

public abstract class Layer {

	String layername;
	float momentum;
	float leraning_rate;
	float gradient;
	float weight_dcay;
	Common_method.FloatFunction<Float> activation;
	Common_method.FloatFunction<Float> dactivation;
	int flatsize;
	String solver;
	//Updater updater;
	Updater.FloatFunction<Float,Float, Float, Float>  updater;

	protected Layer(String actfunc, String updater){

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

		if(updater.equals("sigmoid")){
			this.updater = (float dw, float w, float learning_rate, float momentum)->Updater.Adagrad(dw, w, learning_rate, momentum);
		}else if(updater.equals("tanh")){
			this.updater = (float dw, float w, float learning_rate, float momentum)->Updater.Adagrad(dw, w, learning_rate, momentum);
		}else if(updater.equals("ReLU")){
			this.updater = (float dw, float w, float learning_rate, float momentum)->Updater.SGD(dw, w, learning_rate, momentum);
		}else if(StringUtils.isEmpty(updater)){
			throw new IllegalArgumentException("specify updater function");
		}else{
			throw new IllegalArgumentException("updater function not supported");
		}
	}

}
