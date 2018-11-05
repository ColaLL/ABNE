package FinalModel;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import ModelWith2OrderNorm.BasicUnit;
import StaticVar.Vars;

public class SupervisedAttentionNetwork4Iter {
	
	
	int foldtrain;
	int dimension;
	String network_x="foursquare";
	String network_y="twitter";
	
	
	String relation_file_x="";
	String output_file_x="";
	String embedding_file_x="";
	String embedding_attn_file_x="";
	
	
	String relation_file_y="";
	String output_file_y="";
	
	
	String embedding_file_y="";
	String embedding_attn_file_y="";
	
	HashMap<String,Double> softmax_x=new HashMap<String,Double>();
	HashMap<String,Double> softmax_y=new HashMap<String,Double>();
	
	
	HashMap<String,double[]> part_5_x_frac=new HashMap<String,double[]>();
	HashMap<String,double[]> part_5_y_frac=new HashMap<String,double[]>();
	
	HashMap<String,Double> part_5_x_deno=new HashMap<String,Double>();
	HashMap<String,Double> part_5_y_deno=new HashMap<String,Double>();
		
	String anchor_file="";
	
	double[] parameter_a_x;
	double[] parameter_a_y;
	
	HashMap<String,double[]> embeddings_x=null;
	HashMap<String,HashSet<String>> relations_x=null;
	HashMap<String,double[]> embeddings_y=null;
	HashMap<String,HashSet<String>> relations_y=null;
	
	Random init_rand=new Random(123);

	
	public SupervisedAttentionNetwork4Iter(
			int dimension,
			int foldtrain,
			String relation_file_x,
			String output_file_x,
			String embedding_file_x,
			String embedding_attn_file_x,
			String relation_file_y,
			String output_file_y,
			String embedding_file_y,
			String embedding_attn_file_y,
			String anchor_file) throws IOException
	{
		this.foldtrain=foldtrain;
		this.dimension=dimension;

		parameter_a_x=new double[this.dimension];
		parameter_a_y=new double[this.dimension];
		for(int i=0;i<this.dimension;i++)
		{
			parameter_a_x[i]=init_rand.nextDouble();
			parameter_a_y[i]=init_rand.nextDouble();
		}
		
		this.relation_file_x=relation_file_x;
		this.output_file_x=output_file_x;
		this.embedding_file_x=embedding_file_x;
		this.embedding_attn_file_x=embedding_attn_file_x;
		
		
		this.relation_file_y=relation_file_y;
		this.output_file_y=output_file_y;
		this.embedding_file_y=embedding_file_y;
		this.embedding_attn_file_y=embedding_attn_file_y;
		
		this.anchor_file=anchor_file;
	
		embeddings_x=this.getUserEmbeddingTest(embedding_file_x);
		relations_x=this.getNetworkRelations(relation_file_x);
		
		embeddings_y=this.getUserEmbeddingTest(embedding_file_y);
		relations_y=this.getNetworkRelations(relation_file_y);
		
		
	}
	
	public HashMap<String,double[]> getAttentionNetwork(HashMap<String,double[]> embeddings,
			HashMap<String,HashSet<String>> relations,
			String post_fix,String output_filename,
			String attn_embeddings_file,
			String network,
			int current_epoch,
			int total_epoch) throws IOException
	{
		HashMap<String,double[]> attn_embeddings=new HashMap<String,double[]>();
		HashMap<String,Double> relation_attn_weights=new HashMap<String,Double>();
		for(String key:relations.keySet())
		{
			double deno=0;
			HashSet<String> followees=relations.get(key);
			key=key+"_"+post_fix;			
			double[] embedding_key=embeddings.get(key);
			double[] attn_embedding=new double[embedding_key.length];

			for(String followee:followees)
			{
				followee=followee+"_"+post_fix;
				double[] embedding_followee=embeddings.get(followee);
				double temp_inner=0;
				if(network.equals("x"))
				{
					for(int i=0;i<embedding_key.length;i++)
					{
						temp_inner+=this.parameter_a_x[i]*embedding_key[i]*embedding_followee[i];
					}
				}
				else
				{
					for(int i=0;i<embedding_key.length;i++)
					{
						temp_inner+=this.parameter_a_y[i]*embedding_key[i]*embedding_followee[i];
					}
				}
				deno+=Math.exp(temp_inner);
				
			}
			if(network.equals("x"))
			{
				part_5_x_deno.put(key, deno);
			}
			else
			{
				part_5_y_deno.put(key, deno);
			}
			double[] temp_part_x_frac=new double[this.dimension];
			double[] temp_part_y_frac=new double[this.dimension];
					
			for(String followee:followees)
			{
				followee=followee+"_"+post_fix;
				double[] embedding_followee=embeddings.get(followee);
				double temp_inner=0;
				double[] hi_hk_inner=new double[embedding_followee.length];
				/*for(int i=0;i<embedding_key.length;i++)
				{
					temp_inner+=this.parameter_a_x[i]*embedding_key[i]*embedding_followee[i];
				}*/
				
				double prob=0;
				if(network.equals("x"))
				{
					for(int i=0;i<embedding_key.length;i++)
					{
						temp_inner+=this.parameter_a_x[i]*embedding_key[i]*embedding_followee[i];
						hi_hk_inner[i]=embedding_key[i]*embedding_followee[i];
					}
					prob=Math.exp(temp_inner)/deno;
					for(int i=0;i<embedding_key.length;i++)
					{
						temp_part_x_frac[i]+=hi_hk_inner[i]*Math.exp(temp_inner);
					}
				}
				else
				{
					for(int i=0;i<embedding_key.length;i++)
					{
						temp_inner+=this.parameter_a_y[i]*embedding_key[i]*embedding_followee[i];
						hi_hk_inner[i]=embedding_key[i]*embedding_followee[i];
					}
					prob=Math.exp(temp_inner)/deno;
					for(int i=0;i<embedding_key.length;i++)
					{
						temp_part_y_frac[i]+=hi_hk_inner[i]*Math.exp(temp_inner);
					}
				}
				
				
				relation_attn_weights.put(key.replace("_"+post_fix, "")+" "+followee.replace("_"+post_fix, ""), prob);
				//bw.write(key.replace("_"+post_fix, "")+" "+followee.replace("_"+post_fix, "")+" "+prob+"\n");
				if(network.equals("x"))
				{	
					softmax_x.put(key+"|"+followee,prob);
				}
				else
				{
					softmax_y.put(key+"|"+followee,prob);
				}
				for(int i=0;i<embedding_key.length;i++)
				{
					attn_embedding[i]+=prob*embedding_followee[i];
				}
			}
			
			if(network.equals("x"))
			{
				part_5_x_frac.put(key, temp_part_x_frac);
			}
			else
			{
				part_5_y_frac.put(key, temp_part_y_frac);
			}
			
			attn_embeddings.put(key, attn_embedding);
		}
		
		//Code for the iteration performance
		/*if((current_epoch+1)%50==0)
		{
			String iter_file_name=attn_embeddings_file+".attn_iter."+(current_epoch+1);
			BufferedWriter bw_iter=BasicUnit.writerData(iter_file_name);
			for(String key:attn_embeddings.keySet())
			{
				bw_iter.write(key+" ");
				double[] temp_vec=attn_embeddings.get(key);
				for(int i=0;i<temp_vec.length;i++)
				{
					bw_iter.write(temp_vec[i]+"|");
				}
				bw_iter.write("\n");
			}
			bw_iter.flush();
			bw_iter.close();
		}*/
		
		if((current_epoch+1)==total_epoch)
		{
			BufferedWriter bw=BasicUnit.writerData(output_filename);
			BufferedWriter bw_attn=BasicUnit.writerData(attn_embeddings_file);
			
			
			for(String key:attn_embeddings.keySet())
			{
				bw_attn.write(key+" ");
				double[] temp_vec=attn_embeddings.get(key);
				for(int i=0;i<temp_vec.length;i++)
				{
					bw_attn.write(temp_vec[i]+"|");
				}
				bw_attn.write("\n");
			}
			
			for(String key:relation_attn_weights.keySet())
			{
				bw.write(key+" "+relation_attn_weights.get(key)+"\n");
			}
			bw.flush();
			bw.close();	
			bw_attn.flush();
			bw_attn.close();
		}
		
		for(String key:embeddings.keySet())
		{
			if(!attn_embeddings.containsKey(key))
			{
				attn_embeddings.put(key,embeddings.get(key));
			}
		}
		return attn_embeddings;
	}
	
	public HashMap<String,HashSet<String>> getNetworkRelations(String filename) throws IOException
	{
		HashMap<String,HashSet<String>> answer=new HashMap<String,HashSet<String>>();
		BufferedReader br=BasicUnit.readData(filename);
		String temp_string=br.readLine();
		while(temp_string!=null)
		{
			String[] array=temp_string.split("\\s+");
			if(answer.containsKey(array[0]))
			{
				answer.get(array[0]).add(array[1]);
			}
			else
			{
				HashSet<String> temp_set=new HashSet<String>();
				//temp_set.add(array[0]);
				temp_set.add(array[1]);
				answer.put(array[0], temp_set);
			}
			temp_string=br.readLine();
		}
		br.close();
		return answer;
	}
	
	
	public HashMap<String,double[]> getUserEmbeddingTest(String filename) throws IOException
	{
		HashMap<String,double[]> answer=new HashMap<String,double[]>();
		BufferedReader br=new BufferedReader(new FileReader(filename));
		String temp_string=br.readLine();
		while(temp_string!=null)
		{
			String[] array=temp_string.split("\\s+");
			String name=array[0];
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
	

	
	public HashMap<String,Double> pos_neg_pairSample(HashMap<String,double[]> twitter_embeddings,
			HashMap<String,double[]> foursquare_embeddings,int neg_samples) throws IOException
	{
		HashMap<String,Double> pairs=new HashMap<String,Double>();
		BufferedReader br=BasicUnit.readData(anchor_file);
		String temp_string=br.readLine();
		while(temp_string!=null)
		{
			pairs.put(temp_string+"_twitter|"+temp_string+"_foursquare", 1.0);
			temp_string=br.readLine();
		}
		String[] keys_twitter =  twitter_embeddings.keySet().toArray(new String[0]);
		String[] keys_foursquare = foursquare_embeddings.keySet().toArray(new String[0]);
		Random random = new Random(123);
		int neg_index=0;
		while(neg_index<neg_samples)
		{
			String randomKey_twitter = keys_twitter[random.nextInt(keys_twitter.length)];
			String randomKey_foursquare = keys_foursquare[random.nextInt(keys_foursquare.length)];
			if(randomKey_twitter.replace("_twitter", "").equals(randomKey_foursquare.replace("_foursquare", "")))
			{
				continue;
			}
			pairs.put(randomKey_twitter+"|"+randomKey_foursquare, 0.0);
			neg_index++;
		}
		return pairs;
	}
	
	public void BackWard(HashMap<String,double[]> twitter_embeddings_attn,
			HashMap<String,double[]> foursquare_embeddings_attn) throws IOException  //square loss
	{
		HashMap<String,Double> samples =pos_neg_pairSample(twitter_embeddings_attn,foursquare_embeddings_attn,5000);
		for(String keys:samples.keySet())
		{
			String twitter_user=keys.split("\\|")[0];
			String foursquare_user=keys.split("\\|")[1];
			double label=samples.get(keys);
			double sigmoid_value=0.0;
			double[] twitter_embeddings=twitter_embeddings_attn.get(twitter_user);
			double[] foursquare_embeddings=foursquare_embeddings_attn.get(foursquare_user);
			double[] ori_embedding_foursquare=embeddings_x.get(foursquare_user);
			double[] ori_embedding_twitter=embeddings_y.get(twitter_user);
			
			for(int i=0;i<twitter_embeddings.length;i++)
			{
				sigmoid_value+=twitter_embeddings[i]*foursquare_embeddings[i];
			}
			sigmoid_value=1/(1+Math.exp(-sigmoid_value));
			double part_1=label-sigmoid_value;//sigmoid-y
			double part_2=sigmoid_value*(1-sigmoid_value);//sigmoid(1-sigmoid)
			double[] part_3_x=twitter_embeddings; // h_n^Y
			double[] part_3_y=foursquare_embeddings;// h_i^X			
			double[] part_4_x=new double[twitter_embeddings.length];
			
			
			if(!relations_x.containsKey(foursquare_user.replace("_foursquare", ""))||
					!relations_y.containsKey(twitter_user.replace("_twitter", "")))
			{
				continue;
			}
			if(relations_x.containsKey(foursquare_user.replace("_foursquare", "")))
			{
				HashSet<String> users_x=relations_x.get(foursquare_user.replace("_foursquare", ""));				
				
				for(String followees:users_x)
				{
					followees=followees+"_foursquare";
					double[] tmp_emd_fee=embeddings_x.get(followees);
					double alpha_ij=softmax_x.get(foursquare_user+"|"+followees);
					for(int i=0;i<part_4_x.length;i++)
					{
						part_4_x[i]+=tmp_emd_fee[i]*alpha_ij
								*(tmp_emd_fee[i]*ori_embedding_foursquare[i]
								-part_5_x_frac.get(foursquare_user)[i]/part_5_x_deno.get(foursquare_user)
								);
					}
				}
			}
			
			double[] part_4_y=new double[twitter_embeddings.length];
			
			if(relations_y.containsKey(twitter_user.replace("_twitter", "")))
			{
				HashSet<String> users_y=relations_y.get(twitter_user.replace("_twitter", ""));
				for(String followees:users_y)
				{
					followees=followees+"_twitter";
					double[] tmp_emd_fee=embeddings_y.get(followees);	
					double alpha_ij=softmax_y.get(twitter_user+"|"+followees);
					for(int i=0;i<part_4_y.length;i++)
					{
						part_4_y[i]+=tmp_emd_fee[i]*alpha_ij
								*(tmp_emd_fee[i]*ori_embedding_twitter[i]
								-part_5_y_frac.get(twitter_user)[i]/part_5_y_deno.get(twitter_user)
								);
					}
				}
			}
			
			
			double [] final_part_x=new double[foursquare_embeddings.length];
			double [] final_part_y=new double[twitter_embeddings.length];
			
			for(int i=0;i<final_part_x.length;i++)
			{
				final_part_x[i]=part_1*part_2*part_3_x[i]*part_4_x[i];
				final_part_y[i]=part_1*part_2*part_3_y[i]*part_4_y[i];
				this.parameter_a_x[i]-=final_part_x[i];
				this.parameter_a_y[i]-=final_part_y[i];
			}		
		}
	}
	

	
	public void GoBackward(int current_epoch,int total_epoch) throws IOException
	{
		HashMap<String,double[]> foursquare_embeddings_attn=
				getAttentionNetwork(embeddings_x, relations_x, network_x, 
						output_file_x,embedding_attn_file_x,"x",current_epoch,total_epoch);

		HashMap<String,double[]> twitter_embeddings_attn=
				getAttentionNetwork(embeddings_y, relations_y, network_y, 
						output_file_y,embedding_attn_file_y,"y",current_epoch,total_epoch);
		
		BackWard(twitter_embeddings_attn,foursquare_embeddings_attn);
		if((current_epoch+1)==total_epoch)
		{
			getAttentionNetwork(embeddings_x, relations_x, network_x, 
							output_file_x,embedding_attn_file_x,"x",current_epoch,total_epoch);

			getAttentionNetwork(embeddings_y, relations_y, network_y, 
							output_file_y,embedding_attn_file_y,"y",current_epoch,total_epoch);
		}
		
	}
	
	public static void main(String[] args) throws IOException
	{
	}

}
