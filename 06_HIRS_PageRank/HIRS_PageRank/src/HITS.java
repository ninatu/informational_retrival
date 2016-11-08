import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class HITS extends Configured implements Tool{

    public static class HITSMapper extends Mapper<LongWritable, Text, Text, FSContainer> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // value имееет структуру FSContainer
            String []parts = value.toString().split("\t");
            String url = parts[0];
            Text textUrl = new Text(url);
            String sValue = parts[1];
            String sArray = parts[2].substring(1, parts[2].length() - 1);
            FSContainer cont = new FSContainer(sValue, sArray);
            /* В данном случае FSContainer содержит
            x, array{"H", y1, y2, ... , yn}, где x - значение хабности урла, y1, y2, ... - куда ссылается урл
            или
            x, array{"A", z1, z2, ... , zn}б где x - значение авторитетности урла, z1, z2, ... - кто ссылается на урл
             */
            String A = new String("A");
            String H = new String("H");

            // Если это не первая итерация, рассылаем авторитетность и хабность
            if (cont.getValue() != null) {
                ArrayList<String> array = cont.getArray();
                String ind = array.remove(0);
                ArrayList<String> urls = array;
                Float val = cont.getValue();

                //  "H" - хабность
                if (ind.compareTo(H) == 0) {
                    ArrayList<String> newHub = new ArrayList<>();
                    newHub.add(H);
                    newHub.add(url);
                    FSContainer hub = new FSContainer(val, newHub);
                    for (String y: urls) {
                        context.write(new Text(y), hub);
                    }
                }
                // "A" - авторитетность
                if (ind.compareTo(A) == 0) {
                    ArrayList<String> newAuth = new ArrayList<>();
                    newAuth.add(A);
                    newAuth.add(url);
                    FSContainer authority = new FSContainer(val, newAuth);
                    for (String y: urls) {
                        context.write(new Text(y), authority);
                    }
                }
            } else {
                // Если это первая итерация, инициализируем авторитетность и хабность
                ArrayList<String> urls = cont.getArray();
                Float one = new Float(1);

                //  "H" - хабность
                ArrayList<String> newHub = new ArrayList<>();
                newHub.add(H);
                newHub.add(url);
                FSContainer hub = new FSContainer(new Float(1), newHub);
                for (String y: urls) {
                    context.write(new Text(y), hub);
                }

                // "A" - авторитетность
                Float count = new Float(urls.size());
                ArrayList<String> newAuth = new ArrayList<>();
                newAuth.add(A);
                newAuth.addAll(urls);
                FSContainer authority = new FSContainer(count, newAuth);
                context.write(textUrl, authority);
            }
        }
    }

    protected static class HITSReduser extends Reducer<Text, FSContainer, Text, FSContainer> {
        @Override
        protected void
    }

    @Override
    public int run(String[] strings) throws Exception {
        return 0;
    }
}
