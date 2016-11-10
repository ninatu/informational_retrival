import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.NLineInputFormat;
import org.apache.hadoop.mapreduce.lib.jobcontrol.ControlledJob;
import org.apache.hadoop.mapreduce.lib.jobcontrol.JobControl;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PageRank extends Configured implements Tool {
    protected static final String parametrN = new String("pagerank.n");
    protected static final String parametrAlpha = new String("pagerank.alpha");
    protected static final String parametrLeftRank = new String("pagerank.leftrank");
    protected static float DEFALUT_ALPHA = (float) 0.85;
    protected static float DEFAULT_N = (float) 1239306;
    protected static Text withoutOutLinks = new Text(new String("withoutOutLinks"));

    protected static class PRMapperStep1 extends Mapper<LongWritable, Text, Text, FSContainer> {
        private float N;
        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = new Configuration();
            String strN = conf.get(parametrN);
            if (strN != null) {
                N = Float.valueOf(strN);
            } else {
                N =  DEFAULT_N;
            }
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
                urlPR = cont.getValue(); // если это не первая итерация
            } else {
                urlPR = (float) 1.0 / N; // если это первая итерация берем ранк 1/N
            }
            int count = outUrls.size();
            if (count != 0) {
                Float pr = new Float(urlPR / (float) count);
                FSContainer outCont = new FSContainer(pr, null);
                for(String outUrl: outUrls) {
                    context.write(new Text(outUrl), outCont);
                }
            } else {
                // пересылаем "оставшийся" ранк, в withoutOutLinks
                context.write(withoutOutLinks, new FSContainer(urlPR, null));
            }
        }
    }
    protected static class PRCombinerStep1 extends Reducer<Text, FSContainer, Text, FSContainer> {
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
    protected static class PRReducerStep1 extends Reducer<Text, FSContainer, Text, FSContainer> {
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
            if (key.compareTo(withoutOutLinks) != 0) {
                // пересылаем значение ранка
                outCont.setValue(new Float(sumPR));
                if (outCont.getList() == null) {
                    outCont.setList(new ArrayList<String>());
                }
                context.write(key, outCont);
            } else {
                // записываем значение оставшегося ранка
                (new Configuration()).set(parametrLeftRank, Float.toString(sumPR));
            }

        }
    }
    protected static class PRMapperStep2 extends Mapper<LongWritable, Text, Text, FSContainer> {
        private float N;
        private float alpha;
        private float leftRank;
        @Override
        protected void setup(Context context) throws IOException {
            Configuration conf = new Configuration();
            String strN = conf.get(parametrN);
            String strAlpha = conf.get(parametrAlpha);
            if (strN != null) {
                N = Float.valueOf(strN);
            } else {
                N = DEFAULT_N;
            }
            if (strAlpha != null) {
                alpha = Float.valueOf(strAlpha);
            } else {
                alpha = DEFALUT_ALPHA;
            }
            String sLeftValue = conf.get(parametrLeftRank);
            if (sLeftValue != null) {
                leftRank = Float.valueOf(sLeftValue);
            } else {
                leftRank = 0;
            }
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

            float prevRank = cont.getValue();
            float newRank = alpha * (prevRank + leftRank / N) + (1 - alpha) / N;
            cont.setValue(new Float(newRank));
            context.write(textUrl, cont);
        }
    }
    protected static class PRReducerStep2 extends Reducer<Text, FSContainer, Text, FSContainer> {
        @Override
        protected void reduce(Text key, Iterable<FSContainer> values, Context context) throws IOException, InterruptedException {
            for(FSContainer value: values) {
                context.write(key, value);
            }
        }
    }

    public Job getJobConfStep1(Configuration conf, String input, String output) throws IOException {
        Job job = Job.getInstance(conf);
        job.setJarByClass(PageRank.class);
        job.setJobName(PageRank.class.getCanonicalName());

        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));
        job.getConfiguration().setInt(NLineInputFormat.LINES_PER_MAP, 100000);

        job.setMapperClass(PageRank.PRMapperStep1.class);
        job.setCombinerClass(PageRank.PRCombinerStep1.class);
        job.setReducerClass(PageRank.PRReducerStep1.class);

        job.setInputFormatClass(NLineInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(FSContainer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(FSContainer.class);
        return job;
    }
    public Job getJobConfStep2(Configuration conf, String input, String output) throws IOException {
        Job job = Job.getInstance(conf);
        job.setJarByClass(PageRank.class);
        job.setJobName(PageRank.class.getCanonicalName());

        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));
        job.getConfiguration().setInt(NLineInputFormat.LINES_PER_MAP, 200000);

        job.setMapperClass(PageRank.PRMapperStep2.class);
        job.setReducerClass(PageRank.PRReducerStep2.class);

        job.setInputFormatClass(NLineInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(FSContainer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(FSContainer.class);
        return job;
    }

    @Override
    public int run(String[] args) throws Exception {
        String input = args[0];
        String out_dir = args[1];
        Configuration conf = getConf();

        Integer iterationCount = 50;
        ControlledJob[]steps = new ControlledJob[iterationCount * 2];
        //ControlledJob[]steps = new ControlledJob[iterationCount];

        steps[0] = new ControlledJob(conf);
        steps[0].setJob(getJobConfStep1(conf, input, out_dir + "/it0/step1"));
        steps[1] = new ControlledJob(conf);
        steps[1].setJob(getJobConfStep2(conf, out_dir + "/it0/step1", out_dir + "/it0/step2"));
        for(Integer i = 1; i < iterationCount; i++) {
            String inStepDir = out_dir + "/it" + (new Integer(i - 1)).toString();
            String outStepDir = out_dir + "/it" + i.toString();
            steps[i * 2] = new ControlledJob(conf);
            steps[i * 2].setJob(getJobConfStep1(conf, inStepDir + "/step2", outStepDir + "/step1"));
            steps[i * 2 + 1] = new ControlledJob(conf);
            steps[i * 2 + 1].setJob(getJobConfStep2(conf, outStepDir + "/step1", outStepDir + "/step2"));
        }

        JobControl control = new JobControl(PageRank.class.getCanonicalName());
        for (ControlledJob step: steps) {
            control.addJob(step);
        }

        for (int i = 1; i < iterationCount * 2; i++) {
            steps[i].addDependingJob(steps[i - 1]);
        }

        new Thread(control).start();
        while (!control.allFinished()) {
            System.out.println("Still running...");
            Thread.sleep(10000);
        }
        return control.getFailedJobList().isEmpty() ? 0 : 1;
    }

    static public void main(String[] args) throws Exception {
        int ret = ToolRunner.run(new PageRank(), args);
        System.exit(ret);
    }
}
