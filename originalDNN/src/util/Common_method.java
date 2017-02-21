package util;

import java.util.List;
import java.util.ListIterator;

import Mersenne.Sfmt;

public class Common_method {

	/**
	 * 配列をシャッフルする
	 * */
	public static <T> void array_shuffle(T[] array, Sfmt mt){
		for (int i = 0; i < array.length; i++) {
			int dst = mt.NextInt(array.length);
			T tmp = array[i];
			array[i] = array[dst];
			array[dst] = tmp;
		}
	}


	/**
	 * リストをシャッフルする
	 * */
	public static void list_shuffle(List list, Sfmt mt){
		Object[] array = list.toArray();
		for (int i = 0; i < array.length; i++) {
			int dst = mt.NextInt(array.length);
			Object tmp = array[i];
			array[i] = array[dst];
			array[dst] = tmp;
		}
		ListIterator it = list.listIterator();
		list.clear();
		for (int i=0; i<array.length; i++) {
//			it.next();
//			it.set(array[i]);
			//System.out.println("--------------------------------------------");
			list.add(array[i]);
		}

	}

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

	@FunctionalInterface
	public interface FloatFunction2<R, S,T, U> {

		/**
		 * Applies this function to the given argument.
		 *
		 * @param value the function argument
		 * @return the function result
		 */
		R apply(float dw, float w, float learning_rate, float momentum);
	}
}
