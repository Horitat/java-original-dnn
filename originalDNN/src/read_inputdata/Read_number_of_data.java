package read_inputdata;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import org.apache.commons.lang3.StringUtils;

public class Read_number_of_data {


	/**
	 * テキストに書かれている入力データをカウントする。
	 * これでトレーニングデータとテストデータの数を設定
	 * @param written_datafile 入力データが書かれているテキストファイルへのパスとファイル名
	 * @return データ数
	 */
	static public int count_data(String written_datafile){
		int number = 0;
		try {
			if(!written_datafile.endsWith(".txt")){
				if(!written_datafile.endsWith("\\")){
					written_datafile = written_datafile + "\\data.txt";
				}else{
					written_datafile = written_datafile + "data.txt";
				}
			}
			File file = new File(written_datafile);
			BufferedReader br1 = new BufferedReader(new FileReader(file));

			String str = br1.readLine();

			while(str != null){
				if(StringUtils.isNotBlank(str)){
					number++;
				}
			}

			br1.close();
			if(number <= 0){
				System.out.println("The file["+written_datafile+"] has no data");
				System.out.println("Specify the correct file path");
//				System.err.println("The file["+written_datafile+"] has no data");
//				System.err.println("Specify the correct file path");
				System.exit(1);
			}
		} catch (IOException e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		}
		return number;
	}

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ

		int train_N =count_data("");
		int test_N = count_data("");

	}

}
