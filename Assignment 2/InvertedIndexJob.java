import java.io.IOException;
import java.util.StringTokenizer;
import java.util.Hashtable;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class InvertedIndexJob {

  public static class InvertedIndexMapper
       extends Mapper<LongWritable, Text, Text, Text>{

    private final static Text document_id = new Text();
    private Text word = new Text();

    public void map(LongWritable key, Text value, Context context
                    ) throws IOException, InterruptedException {
    	String[] value_splited = value.toString().split("\n");
    	String id = value_splited[0];
    	StringTokenizer itr = new StringTokenizer(value_splited[1]);

    	document_id.set(id);
      	while (itr.hasMoreTokens()) {
        	word.set(itr.nextToken());
        	context.write(word, document_id);
      	}
    }
  }

  public static class InvertedIndexReducer
       extends Reducer<Text, Text, Text, Text> {
    private Text result = new Text();

    public void reduce(Text key, Iterable<Text> values,
                       Context context
                       ) throws IOException, InterruptedException {
    	Hashtable<String, Integer> map = new Hashtable<String, Integer>();
    	for (Text value : values) {
    		String word = value.toString();
    		map.put(word, map.getOrDefault(word, 0) + 1);
    	}
    	String documents = "";
    	for(String id : map.keySet()) {
		    documents = documents + id + ":" + map.get(id) + " ";		    
		}
		result.set(documents);
		context.write(key, result);
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "Inverted Index");
    job.setJarByClass(InvertedIndexJob.class);

    job.setMapperClass(InvertedIndexMapper.class);
    // job.setCombinerClass(InvertedIndexReducer.class);
    job.setReducerClass(InvertedIndexReducer.class);
    
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);
    
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));

    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}