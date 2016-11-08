import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.*;
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
import java.util.HashSet;
import java.util.Map;

public class LinkGraphCreator extends Configured implements Tool {

    public static class LGCMapper extends Mapper<LongWritable, Text, Text, FSContainer> {
        private Map<Integer, String> urlsMap = new HashMap<>();
        private DefaultCodec codec;
        private Decompressor decompressor;
        private PageDecoder pageDecoder;

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
                    String url =  UrlHandler.toStandart(parts[1]);
                    urlsMap.put(id, url);
                }
            } finally {
                reader.close();
            }
            codec = new DefaultCodec();
            codec.setConf(fileSystem.getConf());
            decompressor = CodecPool.getDecompressor(codec);
            pageDecoder = new PageDecoder(decompressor);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // декодирование html
            String[] parts = value.toString().split("\t");
            Integer htmlId = Integer.valueOf(parts[0]);
            String url = UrlHandler.toStandart(urlsMap.get(htmlId));
            Text urlText = new Text(url);
            String base64string = parts[1];
            String html = pageDecoder.decodeBase64(base64string);

            // получение ссылок из html
            Document doc = Jsoup.parse(html);
            Elements links = doc.select("a[href]");
            HashSet<String> slinks = new HashSet<String>();
            for(Element element: links) {
                String link = element.attr("href").toString();
                if (UrlHandler.checkHost(link)) {
                    link = UrlHandler.toStandart(link);
                    slinks.add(link);
                } else {
                    String absLink = UrlHandler.absoluteUrl(url, link);
                    if (absLink != null && UrlHandler.checkHost(absLink)) {
                        absLink = UrlHandler.toStandart(absLink);
                        slinks.add(absLink);
                    }
                }
            }
            context.write(urlText, new FSContainer(null, new ArrayList<>(slinks)));
        }

        private class PageDecoder {
            private static final int DEFAULT_BUFFER_SIZE = 64 * 1024;
            private byte[] buffer;
            private Text htmlText;
            private Decompressor decompressor;

            public PageDecoder(Decompressor _decomp) throws IOException {
                decompressor = _decomp;
                buffer = new byte[DEFAULT_BUFFER_SIZE];
                htmlText = new Text();
            }

            String decodeBase64(String base64string) throws IOException {
                byte[] gzipHtml = DatatypeConverter.parseBase64Binary(base64string);
                CompressionInputStream compessStream = codec.createInputStream(new ByteArrayInputStream(gzipHtml), decompressor);
                htmlText.clear();
                int readBytes = 0;
                while (true) {
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
        }

        private static class UrlHandler {
            private static String rightHost = new String("lenta.ru");
            private static String pat1From = new String("^https://");
            private static String pat1To = new String("http://");
            private static String pat2From = new String("/+$");
            private static String pat2To = new String("");
            private static String pat3From = new String("\n");
            private static String pat3To = new String("");

            public static String toStandart(String inUrl) {
                String outUrl = inUrl.replaceAll(pat1From, pat1To);
                outUrl = outUrl.replaceAll(pat2From, pat2To);
                outUrl = outUrl.replaceAll(pat3From, pat3To);
                outUrl = outUrl.trim();
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

    public static class LGCReducer extends Reducer<Text, FSContainer, Text, FSContainer> {
        @Override
        protected void reduce(Text key,  Iterable<FSContainer> values, Context context) throws IOException, InterruptedException {
            HashSet<String> links = new HashSet<>();
            for(FSContainer value: values) {
                links.addAll(value.getArray());
            }
            context.write(key, new FSContainer(null, new ArrayList<>(links)));
        }
    }

    public Job getJobConf(String input, String output) throws IOException {
        Job job = Job.getInstance(getConf());
        job.setJarByClass(LinkGraphCreator.class);
        job.setJobName(LinkGraphCreator.class.getCanonicalName());
        TextInputFormat.addInputPath(job, new Path(input));
        TextOutputFormat.setOutputPath(job, new Path(output));
        job.setMapperClass(LGCMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(FSContainer.class);
        job.setReducerClass(LGCReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(FSContainer.class);
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
