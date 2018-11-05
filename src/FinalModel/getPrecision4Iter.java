package FinalModel;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

import ModelWith2OrderNorm.BasicUnit;
import StaticVar.Vars;

public class getPrecision4Iter {
	
	int fold_train;
	
	public getPrecision4Iter(int i)
	{
		this.fold_train=i;
	}
	
	public HashMap<String,double[]> getUserEmbeddingTest(String filename) throws IOException
	{
		HashSet<String> train_anchors=this.getTrainAnchors();
		HashMap<String,double[]> answer=new HashMap<String,double[]>();
		BufferedReader br=new BufferedReader(new FileReader(filename));
		String temp_string=br.readLine();
		while(temp_string!=null)
		{
			String[] array=temp_string.split("\\s+");
			String name=array[0];
			if(train_anchors.contains(name.replace("_twitter", ""))||
					train_anchors.contains(name.replace("_foursquare", "")))
			{
				temp_string=br.readLine();
				continue;
			}
			String[] string_embeddings=array[1].split("\\|");
			double[] temp_embedding=new double[string_embeddings.length];
			double total=0;
			for(int i=0;i<string_embeddings.length;i++)
			{
				double double_tempweight=Double.parseDouble(string_embeddings[i]);
				total+=double_tempweight*
						double_tempweight;
				temp_embedding[i]=double_tempweight;
			}
			double norm_total=Math.sqrt(total);
			for(int i=0;i<temp_embedding.length;i++)
			{
				temp_embedding[i]/=norm_total;
			}
			answer.put(name, temp_embedding);
			temp_string=br.readLine();
		}
		br.close();
		return answer;
	}
	
	
	public HashSet<String> getAnchors() throws IOException
	{
		HashSet<String> anchors_set=new HashSet<String>();
		String anchors_file=
				Vars.twitter_dir+"/twitter_foursquare_groundtruth/groundtruth."+this.fold_train+".foldtrain.test.number";
		BufferedReader br=BasicUnit.readData(anchors_file);
		String temp_string=br.readLine();
		while(temp_string!=null)
		{
			anchors_set.add(temp_string);
			temp_string=br.readLine();
		}
		br.close();
		return anchors_set;
	}
	
	public HashSet<String> getTrainAnchors() throws IOException
	{
		HashSet<String> anchors_set=new HashSet<String>();
		String anchors_file=
				Vars.twitter_dir+"/twitter_foursquare_groundtruth/groundtruth."+this.fold_train+".foldtrain.train.number";
		BufferedReader br=BasicUnit.readData(anchors_file);
		String temp_string=br.readLine();
		while(temp_string!=null)
		{
			anchors_set.add(temp_string);
			temp_string=br.readLine();
		}
		br.close();
		return anchors_set;
	}
	
	public TreeMap<Integer,Integer> getTopSimilarity(int k,String postfix) throws IOException
	{
		TreeMap<Integer,Integer> answer=new TreeMap<Integer,Integer>();
		for(int i=0;i<k;i++)
		{
			answer.put(i+1, 0);
		}
		HashSet<String> anchors=getAnchors();
		String myspace=
				StaticVar.Vars.twitter_dir+
				"/twitter/embeddings/twitter."+postfix;

		String lastfm=
				StaticVar.Vars.twitter_dir+
				"/foursquare/embeddings/foursquare."+postfix;
		HashMap<String,double[]> myspace_embedding=getUserEmbeddingTest(myspace);
		//System.out.println("combined");
		HashMap<String,double[]> lastfm_embedding=getUserEmbeddingTest(lastfm);
		for(String uid:myspace_embedding.keySet())
		{

			if(!anchors.contains(uid.replace("_twitter", "")))
			{
				continue;
			}

			HashMap<String,Double> temp_answer=new HashMap<String,Double>();
			double[] embedding_1=myspace_embedding.get(uid);
			ArrayList<String> user_set=new ArrayList<String>();
			for(String last_id:lastfm_embedding.keySet())
			{
					double[] embedding_2=lastfm_embedding.get(last_id);
					double cosin_value=BasicUnit.getCosinFast(embedding_1,embedding_2);
					temp_answer.put(last_id, cosin_value);
			}
			String lastfm_name=uid.replace("_twitter", "");
			ArrayList<Map.Entry<String, Double>> temp_answer_list=BasicUnit.sortDoubleMap(temp_answer);
			for(int i=0;i<temp_answer_list.size();i++)
			{
				/*if(lastfm_name.equals("datachick"))
				{
					System.out.println(temp_answer_list.get(i).getKey().replace("_foursquare", ""));
				}*/
				if(lastfm_name.equals(temp_answer_list.get(i).getKey().replace("_foursquare", "")))
				{
					
					/*if(i==0)
					{
						System.out.println(lastfm_name);
					}*/
					for(int j:answer.keySet())
					{
						if(i<j)
						{
							int temp_count=answer.get(j)+1;
							answer.put(j, temp_count);
						}
					}
				}
				if(i==k-1)
				{
					break;
				}
			}
		}
		return answer;	
	}
	
	public TreeMap<Integer,Integer> getTopSimilarityReverse(int k,String postfix) throws IOException
	{
		TreeMap<Integer,Integer> answer=new TreeMap<Integer,Integer>();
		for(int i=0;i<k;i++)
		{
			answer.put(i+1, 0);
		}
		HashSet<String> anchors=getAnchors();
		String lastfm=
				StaticVar.Vars.twitter_dir+
				"/twitter/embeddings/twitter."+postfix;
		String myspace=
				StaticVar.Vars.twitter_dir+
				"/foursquare/embeddings/foursquare."+postfix;
		HashMap<String,double[]> myspace_embedding=getUserEmbeddingTest(myspace);
		HashMap<String,double[]> lastfm_embedding=getUserEmbeddingTest(lastfm);
		for(String uid:myspace_embedding.keySet())
		{
			if(!anchors.contains(uid.replace("_foursquare", "")))
			{
				continue;
			}
			HashMap<String,Double> temp_answer=new HashMap<String,Double>();
			double[] embedding_1=myspace_embedding.get(uid);
			ArrayList<String> user_set=new ArrayList<String>();
			for(String last_id:lastfm_embedding.keySet())
			{	
					double[] embedding_2=lastfm_embedding.get(last_id);
					double cosin_value=BasicUnit.getCosinFast(embedding_1,embedding_2);
					temp_answer.put(last_id, cosin_value);	
			}
			ArrayList<Map.Entry<String, Double>> temp_answer_list=BasicUnit.sortDoubleMap(temp_answer);
			String lastfm_name=uid.replace("_foursquare", "");
			int index=0;
			for(int i=0;i<temp_answer_list.size();i++)
			{
				if(lastfm_name.equals(temp_answer_list.get(i).getKey().replace("_twitter", "")))
				{
					/*if(i==0)
					{
						System.out.println(lastfm_name);
					}*/
					for(int j:answer.keySet())
					{
						if(i<j)
						{
							int temp_count=answer.get(j)+1;
							answer.put(j, temp_count);
						}
					}
				}
				if(i==k-1)
				{
					break;
				}
			}
		}
		return answer;
		
	}
	
	public void getFinalAnswer(String embedding_type) throws IOException
	{
		HashSet<String> anchors=getAnchors();
		double total=anchors.size()*2;
		for(int i=10000000;i<=10000000;i+=10000000)
		{
			for(int j=50;j<=500;j+=50)
			{
				String file_postfix=embedding_type+"."+i+".attn_iter."+j;
				TreeMap<Integer, Integer> answer_1=getTopSimilarity(30,file_postfix); //twitter find foursquare
				TreeMap<Integer, Integer> answer_2=getTopSimilarityReverse(30,file_postfix);//foursquare find twitter
				for(int key:answer_1.keySet())
				{
					double right=answer_1.get(key)+answer_2.get(key);
					if(key==1)
						System.out.print(right/total+",");
				}
			}
		}

	}
	
	public static void main(String[] args) throws IOException
	{

		long start = System.currentTimeMillis();
		for(int i=9;i<10;i+=1)
		{
			getPrecision4Iter test=new getPrecision4Iter(i);
			String temp_string="Attn.embedding.2_itercount."+i+".number.100_dim";
			test.getFinalAnswer(temp_string);
			
		}
		long end = System.currentTimeMillis();
		//System.out.println(end-start);
	}

}



