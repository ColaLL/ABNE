package FinalModel;

import java.io.IOException;

import StaticVar.Vars;

public class ABNE {
	
	public static void main(String[] args) throws IOException
	{
		int attention_epoch=500;
		int embeding_epoch=10000000;
		int total_iter=2;
		for(int dimension=100;dimension<=100;dimension+=4)
		{
			for(int train_ratio=9;train_ratio<10;train_ratio++)
			{
				double start = System.currentTimeMillis();
				for(int iter_count=1;iter_count<=total_iter;iter_count++)
				{
					String fileInterop=".number";
					
					String anchor_file=
							Vars.twitter_dir+"twitter_foursquare_groundtruth/groundtruth."+train_ratio+".foldtrain.train"+fileInterop;
					
					String networkx_file=
							StaticVar.Vars.twitter_dir+"/foursquare/following"+fileInterop;
					String ABNE_embedding_x=
							StaticVar.Vars.twitter_dir+
							"/foursquare/embeddings/foursquare.ABNE.embedding."+iter_count+"_itercount."+train_ratio+fileInterop;
					String Attn_embedding_x=
							StaticVar.Vars.twitter_dir+
							"/foursquare/embeddings/foursquare.Attn.embedding."+iter_count+"_itercount."+train_ratio+fileInterop;
					String networkx_attn_file=
							StaticVar.Vars.twitter_dir+"/foursquare/following"+fileInterop+"."+train_ratio+".attn";
					
				
					
					String networky_file=
							StaticVar.Vars.twitter_dir+"/twitter/following"+fileInterop;
					String ABNE_embedding_y=
							StaticVar.Vars.twitter_dir+
							"/twitter/embeddings/twitter.ABNE.embedding."+iter_count+"_itercount."+train_ratio+fileInterop;
					String Attn_embedding_y=
							StaticVar.Vars.twitter_dir+
							"/twitter/embeddings/twitter.Attn.embedding."+iter_count+"_itercount."+train_ratio+fileInterop;
					String networky_attn_file=
							StaticVar.Vars.twitter_dir+"/twitter/following"+fileInterop+"."+train_ratio+".attn";
					
					
					
					if(iter_count==1)
					{
						IONEAttention EmbedingModel=new IONEAttention(dimension,train_ratio,
								anchor_file,
								networkx_file,ABNE_embedding_x,
								networky_file,ABNE_embedding_y);
						EmbedingModel.Train(embeding_epoch);
					}
					else
					{
						IONEAttention EmbedingModel=new IONEAttention(dimension,train_ratio,
								anchor_file,
								networkx_attn_file,ABNE_embedding_x,
								networky_attn_file,ABNE_embedding_y);
						EmbedingModel.Train(embeding_epoch);
					}
					
					ABNE_embedding_x=ABNE_embedding_x+"."+dimension+"_dim"+"."+embeding_epoch;
					ABNE_embedding_y=ABNE_embedding_y+"."+dimension+"_dim"+"."+embeding_epoch;
					
					Attn_embedding_x=Attn_embedding_x+"."+dimension+"_dim"+"."+embeding_epoch;
					Attn_embedding_y=Attn_embedding_y+"."+dimension+"_dim"+"."+embeding_epoch;
					SupervisedAttentionNetwork4Iter AttnModel=new SupervisedAttentionNetwork4Iter(dimension,train_ratio,
							networkx_file,networkx_attn_file,ABNE_embedding_x,Attn_embedding_x,
							networky_file,networky_attn_file,ABNE_embedding_y,Attn_embedding_y,
							anchor_file);
					for(int attn_index=0;attn_index<attention_epoch;attn_index++)
					{
						if(attn_index%10==0)
							System.out.println("attn model running, the iter number is "+attn_index);
						AttnModel.GoBackward(attn_index,attention_epoch);
					}				
				}
				double end = System.currentTimeMillis();
				System.out.println((end-start)/1000.0/60);
			}
		}
	}

}
