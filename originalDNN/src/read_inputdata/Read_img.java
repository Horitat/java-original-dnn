package read_inputdata;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;

import javax.imageio.ImageIO;

public class Read_img {

	/*
	 *後々、4次元配列を返すように扱う
	 *[トレーニングかテストのデータ数][チャンネル][サイズ][サイズ]
	 *その時に、get_files_tostringやディレクトリチェックを入れる
	 *
	 */

	/**
	 * カラー画像を読み取り配列に入れる。チャネルは3固定
	 * @param input 読み取りたい画像が入っているフォルダパス
	 * @return 読み取った画像
	 */
	public static int[][][] read_color_img(String input){
		BufferedImage inputimg;
		try {
			File check = new File(input);

//			if(!check.isDirectory()){
//				System.out.println("Specify path["+input+"] is not directory");
//				System.exit(1);
//			}
			inputimg = ImageIO.read(new File(input));
			int wid = inputimg.getWidth(), hei = inputimg.getHeight();
			int[] pixel = inputimg.getRGB(0,0,wid,hei,null,0,wid);

			int[][][] picture = new int[3][wid][hei];
			//System.out.println(pixel.length);
			for(int i=0; i<wid; i++){
				int n = hei * i;
				for(int j=0; j<hei; j++){
					int argb = pixel[n+j];

					picture[0][i][j] = argb >> 16 &0xFF;
					picture[1][i][j] = argb >> 8 &0xFF;
					picture[2][i][j] = argb >> 0 &0xFF;
					//System.out.println((n+j)+",");
				}
			}
			//System.out.println(picture[0][0].length);

			return picture;
		} catch (IOException e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		}
		return null;
	}


	/**
	 * カラー画像を読み取り配列に入れる。チャネルは3固定
	 * @param input 読み取りたい画像が入っているフォルダパス
	 * @return 読み取った画像
	 */
	public static int[][][] read_color_img_to_gray(String input){
		BufferedImage inputimg;
		try {
			File check = new File(input);

//			if(!check.isDirectory()){
//				System.out.println("Specify path["+input+"] is not directory");
//				System.exit(1);
//			}
			inputimg = ImageIO.read(new File(input));
			int wid = inputimg.getWidth(), hei = inputimg.getHeight();

			int[][][] picture = new int[1][wid][hei];
			//System.out.println(pixel.length);
			for(int i=0; i<wid; i++){
				for(int j=0; j<hei; j++){
					int argb = inputimg.getRGB(i,j);

					picture[0][i][j] = (int)(0.299*(argb >> 16 &0xFF) + 0.587 * (argb >> 8 & 0xff) + 0.114 * (argb & 0xff) + 0.5);
					//System.out.println((n+j)+",");
				}
			}
			System.out.println(wid+":"+hei);
			System.out.println(picture[0].length+":"+picture[0][0].length);

			return picture;
		} catch (IOException e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		}
		return null;
	}


	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		String input_folderpath = "";
		String input_testfilepath = "", input_trainfilepath = "";

		//テスト用
//		read_color_img("C:\\pleiades\\workspace\\conv_output.jpg");
		read_color_img_to_gray("C:\\pleiades\\workspace\\conv_output.jpg");
		//************

		/*
		switch(args.length){
		case 0:
			input_folderpath = "C:\\Users\\WinGAIA\\Desktop\\割れ写真判定学習用512\\まとめ_bk\\傷汚れなし\\";
			input_testfilepath = input_folderpath + "test_data\\";
			input_trainfilepath = input_folderpath + "train_data\\";
			break;

		case 1:
			input_folderpath = args[1];
			input_testfilepath = input_folderpath + "test_data\\";
			input_trainfilepath = input_folderpath + "train_data\\";
			break;

		case 3:
			input_folderpath = args[1];
			input_testfilepath = input_folderpath + args[2];
			input_trainfilepath = input_folderpath + args[3];

		default:
			System.err.println("引数でフォルダを指定してください");
		}


		int[][][] traindata = read_color_img(input_trainfilepath);
		int[][][] testdata = read_color_img(input_testfilepath);
		*/
	}

	/**
	 * フォルダにあるjpgファイルを獲得する
	 * @param folderpath フォルダパス
	 * @return ファイル一覧
	 */
	public static String[] get_files_tostring(String folderpath){
		File files = new File(folderpath);

		if(!files.isDirectory()){
			System.out.println("getting is File!!" + folderpath);
		}
		//ファイル一覧を取得、取得はファイル名＋拡張子のみ
		return files.list(new Filter_TEST());
	}
}

	class Filter_TEST implements FilenameFilter{
		public boolean accept(File dir, String name){

			if(name.matches(".*jpg$") || name.matches(".*JPG$")){
				return true;
			}
			//System.out.println(name);
			return false;
		}
	}

