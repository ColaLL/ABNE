package FinalModel;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;

import ModelWith2OrderNorm.BasicUnit;
import StaticVar.Vars;

public class IONEAttention {
	
	//network x as the foursquare, network y as the twitter 
	
	int foldtrain=0;
	String anchor_file="";
	String networkx_file="";
	String networky_file="";
	String ouput_filename_networkx="";
	String ouput_filename_networky="";
	int dimension=100;
	
	public IONEAttention(int dimension,int foldtrain,String anchor_file,
			String networkx_file,String ouput_filename_networkx,
			String networky_file,String ouput_filename_networky)
	{
		this.dimension=dimension;
		this.foldtrain=foldtrain;
		this.anchor_file=anchor_file;
		
		this.networkx_file=networkx_file;
		this.ouput_filename_networkx=ouput_filename_networkx;
		
		this.networky_file=networky_file;
		this.ouput_filename_networky=ouput_filename_networky;
	}
	
	public HashMap<String,String> getNetworkAnchors(String postfix_1,String postfix_2) throws IOException
	{
		HashMap<String,String> answer_map=new HashMap<String,String>();
		BufferedReader br=BasicUnit.readData(anchor_file);
		String temp_string=br.readLine();
		while(temp_string!=null)
		{
			answer_map.put(temp_string+postfix_1, temp_string+postfix_2);
			temp_string=br.readLine();
		}
		return answer_map;
	}
	
	
	public void Train(int total_iter) throws IOException
	{
		/*String run_count="attn";
		String networkx_file=
				StaticVar.Vars.twitter_dir+"/foursquare/following"+fileInterop;
		String ouput_filename_networkx=
				StaticVar.Vars.twitter_dir+
				"/foursquare/embeddings/foursquare.embedding.update.2SameAnchor.twodirectionContext."+run_count+"_runcount."+this.foldtrain+fileInterop;
		
		String networky_file=
				StaticVar.Vars.twitter_dir+"/twitter/following"+fileInterop;
		String ouput_filename_networky=
				StaticVar.Vars.twitter_dir+
				"/twitter/embeddings/twitter.embedding.update.2SameAnchor.twodirectionContext."+run_count+"_runcount."+this.foldtrain+fileInterop;
		*/
		
		
		IONEAttentionUpdate twoOrder_foursquare=
				new IONEAttentionUpdate(this.dimension,networkx_file,"foursquare");
		twoOrder_foursquare.init();
		
		IONEAttentionUpdate twoOrder_twitter=
				new IONEAttentionUpdate(this.dimension,networky_file,"twitter");
		twoOrder_twitter.init();
		
		HashMap<String,String> foursquare_twitter_anchor=getNetworkAnchors("_foursquare","_twitter");
		HashMap<String,String> twitter_foursquare_anchor=getNetworkAnchors("_twitter","_foursquare");
		System.out.println(foursquare_twitter_anchor.size());
		System.out.println(twitter_foursquare_anchor.size());
		
		for(int i=0;i<total_iter;i++)
		{
			//TwoOrder.Train(i, total_iter);
			twoOrder_foursquare.Train(i, total_iter, twoOrder_foursquare.answer,
					twoOrder_foursquare.answer_context_input,
					twoOrder_foursquare.answer_context_output,
					foursquare_twitter_anchor);
			twoOrder_twitter.Train(i, total_iter, twoOrder_foursquare.answer,
					twoOrder_foursquare.answer_context_input,
					twoOrder_foursquare.answer_context_output,
					twitter_foursquare_anchor);
			if((i+1)%10000000==0)
			{
				String output_file_foursquare=ouput_filename_networkx+"."+this.dimension+"_dim"+"."+(i+1);
				twoOrder_foursquare.output_ori(output_file_foursquare);
				
				String output_file_twitter=ouput_filename_networky+"."+this.dimension+"_dim"+"."+(i+1);
				twoOrder_twitter.output(output_file_twitter,
						twitter_foursquare_anchor,
						twoOrder_foursquare.answer);				
			}
		}
	}
	public static void main(String[] args) throws IOException
	{
		
	}
	

}
