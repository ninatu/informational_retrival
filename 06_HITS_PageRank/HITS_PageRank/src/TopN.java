import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.LineReader;

import java.io.IOException;
import java.util.*;

public class TopN {
    private static SortedMap<Float,String> getTopN(LineReader reader, Integer N, boolean isHits) throws IOException
    {
        TreeMap<Float,String> sortedPairs = new TreeMap<>(new Comparator<Float>()
        {
            @Override
            public int compare(Float o1, Float o2) {
                return o2.compareTo(o1);
            }
        });
        Text line = new Text();
        while (true) {
            line.clear();
            int readBytes = reader.readLine(line);
            if (readBytes <= 0) {
                break;
            }
            String[] parts = line.toString().split("\t");
            if (isHits) {
                FSContainer cont = new FSContainer(parts[1], parts[2]);
                if (cont.getList().get(0) == "A") {
                    sortedPairs.put(cont.getValue(), parts[0]);
                }
            } else {
                sortedPairs.put(Float.valueOf(parts[1]), parts[0]);
            }
        }
        // compute N-th key
        int curCount = 0;
        Float nthKey = new Float(1);
        for (Map.Entry<Float,String> entry: sortedPairs.entrySet()) {
            if (curCount == N) {
                nthKey = entry.getKey();
                break;
            }
            curCount++;
        }
        return sortedPairs.headMap(nthKey);
    }


    public static void main(String[] args) throws IOException {
        if (args.length != 4 || args[2].compareTo("HITS") != 0 && args[2].compareTo("PageRank") != 0) {
            System.err.println("Usage args: path_to_ranks path_to_urls HITS/PageRank N");
            return;
        }
        String ranksName = args[0];
        String urlsName = args[1];
        String alg = args[2];
        Integer N = Integer.valueOf(args[3]);
        boolean isHist = (alg == "HITS") ? true : false;

        FileSystem fileSystem = FileSystem.get(new Configuration());
        LineReader rankReader = new LineReader(fileSystem.open(new Path(ranksName)));
        SortedMap<Float, String> sortedTop = getTopN(rankReader, N, isHist);
        rankReader.close();

        Set<String> topNumbers = new HashSet<>(sortedTop.values());
        Map<String, String> urlsMap = new TreeMap<>();
        LineReader urlsReader = new LineReader(fileSystem.open(new Path(urlsName)));
        Text line = new Text();
        // считываем строку в которой записано кол-во
        urlsReader.readLine(line);
        while (true) {
            line.clear();
            int readBytes = urlsReader.readLine(line);
            if (readBytes <= 0) {
                break;
            }
            String[] parts = line.toString().split("\t");
            if (topNumbers.contains(parts[0])) {
                urlsMap.put(parts[0], parts[1]);
            }
        }
        urlsReader.close();
        for(Map.Entry<Float, String> entry: sortedTop.entrySet()) {
            System.out.println(urlsMap.get(entry.getValue()) + " " + entry.getKey());
        }
    }
}
