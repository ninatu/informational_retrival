import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class PageRank {

    protected static class PRMapper extends Mapper<LongWritable, Text, Text, FSContainer> {
        @Override
        protected void map(LongWritable key, Text value, Context context) {
            // value имееет структуру FSContainer
            String []parts = value.toString().split("\t");
            String url = parts[0];
            Text textUrl = new Text(url);
            String sValue = parts[1];
            String sArray = parts[2];
            FSContainer cont = new FSContainer(sValue, sArray);

        }
    }

}
