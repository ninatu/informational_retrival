import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.LineReader;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class LinksNumerator {

    private static HashMap<String, Integer> numerateLinks(LineReader reader, BufferedWriter writer) throws IOException {
        HashMap<String, Integer> urlMap = new HashMap<>();
        Integer count = new Integer(0);
        String linksFormat = new String("%s\t%s\n");
        Text line = new Text();
        Integer lineCount = new Integer(-1);
        while (true) {
            line.clear();
            int readBytes = reader.readLine(line);
            lineCount++;
            if (readBytes <= 0) {
                break;
            }
            String parts[] = line.toString().split("\t");
            if (parts.length != 3) {
                System.err.println(lineCount.toString() + " " + line.toString());
                continue;
            }
            String inUrl = parts[0];
            String sValue = parts[1];
            String sArray = parts[2];
            FSContainer cont = new FSContainer(sValue, sArray);
            ArrayList<String> outUrls = cont.getArray();
            Integer nInUrl = urlMap.get(inUrl);
            if (nInUrl == null) {
                nInUrl = new Integer(count);
                urlMap.put(inUrl, nInUrl);
                count ++;
            }
            ArrayList<String> numbsUrls = new ArrayList<>();
            for(String outUrl: outUrls) {
                Integer nOutUrl = urlMap.get(outUrl);
                if (nOutUrl == null) {
                    nOutUrl = new Integer(count);
                    urlMap.put(outUrl, nOutUrl);
                    count ++;
                }
                numbsUrls.add(nOutUrl.toString());
            }
            writer.write(String.format(linksFormat, nInUrl.toString(),
                    (new FSContainer(null, numbsUrls)).toString()));
        }
        reader.close();
        writer.close();
        return urlMap;
    }


    public static void main(String[] args) throws IOException {
        String input = args[0];
        FileSystem fileSystem = FileSystem.get(new Configuration());
        Path inPath = new Path(input);
        Path urlsPath = Path.mergePaths(inPath.getParent(), new Path("/urls"));
        Path linksPath = Path.mergePaths(inPath.getParent(), new Path("/links"));
        LineReader inReader = new LineReader(fileSystem.open(inPath));
        BufferedWriter linksWriter = new BufferedWriter(new OutputStreamWriter(fileSystem.create(linksPath)));
        BufferedWriter urlsWriter = new BufferedWriter(new OutputStreamWriter(fileSystem.create(urlsPath)));

        Map<String, Integer> urlsMap = numerateLinks(inReader, linksWriter);
        String urlsFormat = new String("%s\t%s\n");
        urlsWriter.write(Integer.toString(urlsMap.size()) + "\n");
        for (Map.Entry<String, Integer> entry : urlsMap.entrySet()) {
            urlsWriter.write(String.format(urlsFormat, entry.getValue().toString(), entry.getKey()));
        }
        urlsWriter.close();
    }
}
