import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.LineReader;

import java.io.IOException;

public class GraphStatistics {
    public static void main(String[] args) throws IOException {
        String input = args[0];
        FileSystem fileSystem = FileSystem.get(new Configuration());

        LineReader reader = new LineReader(fileSystem.open(new Path(input)));

        Text line = new Text();
        Integer allCount = 0;
        Integer withoutLinks = 0;
        while (true) {
            line.clear();
            int readBytes = reader.readLine(line);
            if (readBytes <= 0) {
                break;
            }
            String[] parts = line.toString().split("\t");
            FSContainer cont = new FSContainer(parts[1], parts[2]);
            allCount ++;
            if (cont.getList().size() == 0) {
                withoutLinks++;
            }
        }
        System.out.println("Count: " + allCount);
        System.out.println("WithoutLinks: " + withoutLinks);
        System.out.println("Persent without links: " + (new Float(withoutLinks) / new Float(allCount)));
    }
}
