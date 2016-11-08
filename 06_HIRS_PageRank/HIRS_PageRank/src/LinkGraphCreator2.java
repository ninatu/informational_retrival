import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.server.common.JspHelper;
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

public class LinkGraphCreator2 extends Configured implements Tool {


    public static class LGCMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
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
                    context.write(new IntWritable(htmlId), new Text(id.toString() + " " + link));
                } else {
                    context.write(new IntWritable(htmlId), new Text(link));
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

    public static class LGCReducer extends Reducer<IntWritable, Text, IntWritable,  Text> {
        @Override
        protected void reduce(IntWritable key,  Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for(Text value: values) {
                context.write(key, value);
            }

        }
    }

    public Job getJobConf(String input, String output) throws IOException {
        Job job = Job.getInstance(getConf());
        job.setJarByClass(LinkGraphCreator2.class);
        job.setJobName(LinkGraphCreator2.class.getCanonicalName());
        TextInputFormat.addInputPath(job, new Path(input));
        TextOutputFormat.setOutputPath(job, new Path(output));
        job.setMapperClass(LGCMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setReducerClass(LGCReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = getJobConf(args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public void help(String[] args) throws Exception {
        int ret = ToolRunner.run(new LinkGraphCreator2(), args);
        System.exit(ret);
    }


    static String urlName = new String("./urls.txt");
    static String dataName = new String("./docs-000.txt");
    public static void help2(String[] args) throws IOException, URISyntaxException {
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
    public static void main(String[] args) throws MalformedURLException {
        String xurl = new String("https://age.lenta.ru/news/2013/01/02/enc/?height=450&iframe=true&width=900////");

        String relPath = new String("https://blabla.ru/rubrics/culture/");
        String surl = UrlHandler.toStandart(xurl);
        URL url = new URL(surl);
        System.err.println(url.getHost());
        System.err.println(surl);
        System.err.println(UrlHandler.checkHost(surl));
        System.err.println(UrlHandler.checkHost(relPath));
        String absUrl = UrlHandler.absoluteUrl(surl, relPath);
        System.err.println(UrlHandler.toStandart(absUrl));

    }
    private static class UrlHandler {
        private static String rightHost = new String("lenta.ru");
        private static String pat1From = new String("^https://");
        private static String pat1To = new String("http://");
        private static String pat2From = new String("/+$");
        private static String pat2To = new String("");
        public static String toStandart(String inUrl) {
            String outUrl = inUrl.replaceAll(pat1From, pat1To);
            outUrl = outUrl.replaceAll(pat2From, pat2To);
            return outUrl;
        }
        public static String absoluteUrl(String parUrl, String relUrl) {
            try {
                URL url = new URL(new URL(parUrl), relUrl);
                return url.toString();
            } catch (MalformedURLException e) {
                return null;
            }
        }
        public static boolean checkHost(String inUrl) {
            try {
                URL url = new URL(inUrl);
                if (url.getHost().compareTo(rightHost) == 0) {
                    return true;
                } else {
                    return false;
                }
            } catch (MalformedURLException e) {
                return false;
            }
        }

    }

}
