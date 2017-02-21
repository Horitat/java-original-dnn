package util;

import java.util.Random;

import Mersenne.Sfmt;

public class RandomGenerator {
	/**
	 *
	 * @param min 発生させたい乱数の最小値
	 * @param max 発生させたい乱数の最大値
	 * @param mt メルセンヌツイスターのインスタンス
	 * @return 指定範囲内の乱数
	 */
	public static float uniform(double min, double max, Sfmt mt) {
		return (float) (mt.NextUnif() * (max - min) + min);
	}

	/**
	 *
	 * @param min 発生させたい乱数の最小値
	 * @param max 発生させたい乱数の最大値
	 * @param mt メルセンヌツイスターのインスタンス
	 * @return 指定範囲内の乱数
	 */
	public static float uniform(float min, float max, Sfmt mt) {
		return (float)mt.NextUnif() * (max - min) + min;
	}

	public static float uniform(float min, float max, Random rn) {
		// TODO 自動生成されたメソッド・スタブ
		return (float)rn.nextFloat() * (max - min) + min;
	}

	/**
	 *
	 * @param n 繰り返し回数
	 * @param p 0より大きく1未満の実数値
	 * @param mt メルセンヌツイスターのインスタンス
	 * @return pが出た回数
	 */
	public static int binomial(int n, double p, Sfmt mt) {
		if(p < 0. || p > 1.) return 0;
		//System.out.println(p);
		int c = 0;
		double r;

		for(int i=0; i<n; i++) {
			r = mt.NextUnif();
			if (r < p) c++;
		}

		return c;
	}

	/**
	 *
	 * @param n 繰り返し回数
	 * @param p 0より大きく1未満の実数値
	 * @param mt メルセンヌツイスターのインスタンス
	 * @return pが出た回数
	 */
	public static int binomial(int n, float p, Sfmt mt) {
		if(p < 0.f || p > 1.f) return 0;
		//System.out.println(p);
		int c = 0;
		double r;

		for(int i=0; i<n; i++) {
			r = mt.NextUnif();
			if (r < p) c++;
		}

		return c;
	}

	/**
	 * xavierでの初期化
	 * @param n ノード数
	 * @param mt メルセンヌツイスターのインスタンス
	 * @return (0,1/√n)に基づく正規分布
	 */
	public static float xavier (int n, Sfmt mt){
		//nの行列か配列を作り、返すように作るか
		return (float)(mt.NextNormal() / Math.sqrt(n));
	}

	/**
	 * Heでの初期化
	 * @param n ノード数
	 * @param mt メルセンヌツイスターのインスタンス
	 * @return (0,1/√2/n)に基づく正規分布
	 */
	public static float He(int n, Sfmt mt){
		return (float)(mt.NextNormal() / Math.sqrt(2./n) );
	}
}
