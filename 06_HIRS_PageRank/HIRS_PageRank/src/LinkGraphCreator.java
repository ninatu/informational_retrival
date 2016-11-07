import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.CodecPool;
import org.apache.hadoop.io.compress.CompressionInputStream;
import org.apache.hadoop.io.compress.Decompressor;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.LineReader;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import javax.xml.bind.DatatypeConverter;
import java.io.ByteArrayInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class LinkGraphCreator extends Configured implements Tool {


    public static class LGCMapper extends Mapper<LongWritable, Text, IntWritable, IntWritable> {
        private static final int DEFAULT_BUFFER_SIZE = 64 * 1024;
        private Map<String, Integer> urlsMap = new HashMap<>();
        private GzipCodec codec;
        private Decompressor decompressor;
        private Text htmlText;
        private byte[] buffer;
        private static String rightHost = new String("lenta.ru");


        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            FileSystem fileSystem = FileSystem.get(new Configuration());
            FileSplit split = (FileSplit) context.getInputSplit();
            Path urlsPath = Path.mergePaths(split.getPath().getParent(), new Path("/urls.txt"));
            LineReader reader = new LineReader(fileSystem.open(urlsPath));
            Text line = new Text();
            try {
                while (true) {
                    int count = reader.readLine(line);
                    if (count <= 0) {
                        break;
                    }
                    String parts[] = line.toString().split("\t");
                    int id = Integer.valueOf(parts[0]);
                    String path = (new URL(parts[1])).getPath();
                    urlsMap.put(path, id);
                }
            } finally {
                reader.close();
            }
            codec = new GzipCodec();
            codec.setConf(fileSystem.getConf());
            decompressor = CodecPool.getDecompressor(codec);
            buffer = new byte[DEFAULT_BUFFER_SIZE];
            htmlText = new Text();
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // декодирование html
            String[] parts = value.toString().split("\t");
            Integer htmlId = Integer.valueOf(parts[0]);
            String base64string = parts[1];
            String html = decodeBase64(base64string);

            // получение ссылок из html
            Document doc = Jsoup.parse(html);
            Elements links = doc.select("a[href]");
            for(Element element: links) {
                String link = element.attr("href").toString();
                Integer id = findLink(link);
                if (id != null) {
                    context.write(new IntWritable(htmlId), new IntWritable(id));
                }
            }
        }

        private String decodeBase64(String base64string) throws IOException {
            byte[] gzipHtml = DatatypeConverter.parseBase64Binary(base64string);
            CompressionInputStream compessStream = codec.createInputStream(new ByteArrayInputStream(gzipHtml), decompressor);
            htmlText.clear();
            int  readBytes = 0;
            while(true) {
                try {
                    readBytes = compessStream.read(buffer);
                    if (readBytes <= 0)
                        break; // EOF
                    htmlText.append(buffer, 0, readBytes);
                } catch (EOFException eof) {
                    break;
                }
            }
            compessStream.close();
            return htmlText.toString();
        }

        private Integer findLink(String link) {
            try {
                URL linkUrl = new URL(link);
                if (linkUrl.getHost() == rightHost) {
                    return urlsMap.get(linkUrl.getPath());
                }
            } finally {
                return urlsMap.get(link);
            }
        }
    }

    public static class LGCReducer extends Reducer<IntWritable, IntWritable, IntWritable,  NumberContainer> {
        @Override
        protected void reduce(IntWritable key,  Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            ArrayList<Integer> array = new ArrayList<>();
            for(IntWritable value: values) {
                array.add(new Integer(value.get()));
            }
            context.write(key, new NumberContainer(null, array));
        }
    }

    public Job getJobConf(String input, String output) throws IOException {
        Job job = Job.getInstance(getConf());
        job.setJarByClass(LinkGraphCreator.class);
        job.setJobName(LinkGraphCreator.class.getCanonicalName());
        TextInputFormat.addInputPath(job, new Path(input));
        TextOutputFormat.setOutputPath(job, new Path(output));
        job.setMapperClass(LGCMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setReducerClass(LGCReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(NumberContainer.class);
        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new LinkGraphCreator(), args);
        System.exit(ret);
    }


    static String urlName = new String("./urls.txt");
    static String dataName = new String("./docs-000.txt");
    public static void help(String[] args) throws IOException, URISyntaxException {
        Path dataPath = new Path(dataName);
        FileSystem fileSystem = FileSystem.get(new Configuration());
        LineReader reader = new LineReader(fileSystem.open(dataPath));

        GzipCodec codec = new GzipCodec();
        codec.setConf(fileSystem.getConf());
        Decompressor decompressor = CodecPool.getDecompressor(codec);

        Text text = new Text();
        reader.readLine(text);
        String parts[] = text.toString().split("\t");
        String base64string = parts[1];

        byte[] gzipHtml = DatatypeConverter.parseBase64Binary(base64string);
        CompressionInputStream compessStream= codec.createInputStream(new ByteArrayInputStream(gzipHtml), decompressor);

        byte[] buffer = new byte[64 * 1024];
        Text html = new Text();
        int  readBytes = 0;
        while(true) {
            try {
                readBytes = compessStream.read(buffer);
                if (readBytes <= 0)
                    break; // EOF
                html.append(buffer, 0, readBytes);
            } catch (EOFException eof) {
                break;
            }
        }
        compessStream.close();

        Document doc = Jsoup.parse(html.toString());
        Elements links = doc.select("a[href]");
        for(Element link :links) {
            System.err.println(link.attr("href").toString());
        }
        System.err.println(html.toString());
    }
}
