import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.jobcontrol.ControlledJob;
import org.apache.hadoop.mapreduce.lib.jobcontrol.JobControl;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.LineReader;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PageRank extends Configured implements Tool {
    private static float alpha = (float) 0.85;
    private static float N;

    protected static class PRMapper extends Mapper<LongWritable, Text, Text, FSContainer> {

        @Override
        protected void setup(Context context) throws IOException {
            FileSystem fileSystem = FileSystem.get(new Configuration());
            FileSplit split = (FileSplit) context.getInputSplit();
            Path urlsPath = Path.mergePaths(split.getPath().getParent(), new Path("/urls.txt"));
            LineReader reader = new LineReader(fileSystem.open(urlsPath));
            Text line = new Text();
            reader.readLine(line);
            N = Float.valueOf(line.toString());
            reader.close();
        }
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // value имееет структуру FSContainer
            String []parts = value.toString().split("\t");
            String url = parts[0];
            Text textUrl = new Text(url);
            String sValue = parts[1];
            String sArray = parts[2];
            FSContainer cont = new FSContainer(sValue, sArray);

            // Пересылаем структуру графа
            context.write(textUrl, new FSContainer(null, cont.getList()));

            // Рассылаем ранк
            List<String> outUrls = cont.getList();
            float urlPR;
            if (cont.getValue() != null) {
                urlPR = cont.getValue(); // если это не первая итерация, рассылаем ранк
            } else {
                urlPR = (float) 1.0 / N; // если это первая итерация. рассылаем ранк 1/N
            }
            int count = outUrls.size();
            if (count != 0) {
                Float pr = new Float(urlPR / (float) count);
                FSContainer outCont = new FSContainer(pr, null);
                for(String outUrl: outUrls) {
                    context.write(new Text(outUrl), outCont);
                }
            }
        }
    }
    protected static class PRReducer extends Reducer<Text, FSContainer, Text, FSContainer> {
        @Override
        protected void reduce(Text key, Iterable<FSContainer> values, Context context) throws IOException, InterruptedException {
            FSContainer outCont = new FSContainer();
            float sumPR = 0;
            for(FSContainer value: values) {
                if (value.getValue() != null) {
                    sumPR += value.getValue();
                }
                if(value.getList() != null) {
                    outCont.setList(new ArrayList<>(value.getList()));
                }
            }
            float newPr = alpha * sumPR + (1 - alpha) * (1 / N);
            outCont.setValue(new Float(newPr));
            context.write(key, outCont);
        }
    }

    protected static class PRCombiner extends Reducer<Text, FSContainer, Text, FSContainer> {
        @Override
        protected void reduce(Text key, Iterable<FSContainer> values, Context context) throws IOException, InterruptedException {
            FSContainer outCont = new FSContainer();
            float sumPR = 0;
            for(FSContainer value: values) {
                if (value.getValue() != null) {
                    sumPR += value.getValue();
                }
                if(value.getList() != null) {
                    outCont.setList(new ArrayList<>(value.getList()));
                }
            }
            outCont.setValue(new Float(sumPR));
            context.write(key, outCont);
        }
    }

    public Job getJobConf(String input, String output) throws IOException {

        Job job = Job.getInstance(getConf());
        job.setJarByClass(PageRank.class);
        job.setJobName(PageRank.class.getCanonicalName());
        TextInputFormat.addInputPath(job, new Path(input));
        TextOutputFormat.setOutputPath(job, new Path(output));
        job.setMapperClass(PageRank.PRMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(FSContainer.class);
        job.setReducerClass(PageRank.PRReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(FSContainer.class);
        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        String input = args[0];
        String out_dir = args[1];

        Integer iterationCount = 5;
        ControlledJob[]steps = new ControlledJob[5];

        steps[0] = new ControlledJob(getConf());
        steps[0].setJob(getJobConf(input, out_dir + "/step0"));
        for(Integer i = 1; i < iterationCount; i++) {
            String inStepDir = out_dir + "/step" + (new Integer(i - 1)).toString();
            String outStepDir = out_dir + "/step" + i.toString();
            steps[i] = new ControlledJob(getConf());
            steps[i].setJob(getJobConf(inStepDir, outStepDir));
        }

        JobControl control = new JobControl(HITS.class.getCanonicalName());
        for (ControlledJob step: steps) {
            control.addJob(step);
        }
        for (int i = 1; i < iterationCount; i++) {
            steps[i].addDependingJob(steps[i - 1]);
        }

        new Thread(control).start();
        while (!control.allFinished()) {
            System.out.println("Still running...");
            Thread.sleep(2000);
        }
        return control.getFailedJobList().isEmpty() ? 0 : 1;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new HITS(), args);
        System.exit(ret);
    }
}
