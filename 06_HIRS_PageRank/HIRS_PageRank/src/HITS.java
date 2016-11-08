import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.jobcontrol.ControlledJob;
import org.apache.hadoop.mapreduce.lib.jobcontrol.JobControl;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.ArrayList;
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
            String sArray = parts[2];
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
                List<String> array = cont.getList();
                String ind = array.remove(0);
                List<String> urls;
                urls = array;
                Float val = cont.getValue();

                //  "H" - хабность
                if (ind.compareTo(H) == 0) {
                    List<String> newHub = new ArrayList<>();
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
                List<String> urls = cont.getList();
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
        protected void reduce(Text key, Iterable<FSContainer> values, Context context) throws IOException, InterruptedException {
            float auth = 0;
            float hubth = 0;
            ArrayList<String> authList = new ArrayList<>();
            ArrayList<String> hubList = new ArrayList<>();
            String A = new String("A");
            String H = new String("H");
            authList.add(A);
            hubList.add(H);

            for (FSContainer value: values) {
                List<String> array = value.getList();
                String ind = array.remove(0);
                if (ind.compareTo(A) == 0) {
                    hubth += value.getValue();
                    hubList.addAll(array);
                }
                if (ind.compareTo(H) == 0) {
                    auth += value.getValue();
                    authList.addAll(array);
                }
            }
            context.write(key, new FSContainer(new Float(auth), authList));
            context.write(key, new FSContainer(new Float(hubth), hubList));
        }
    }

    public Job getJobConf(String input, String output) throws IOException {
        Job job = Job.getInstance(getConf());
        job.setJarByClass(HITS.class);
        job.setJobName(HITS.class.getCanonicalName());
        TextInputFormat.addInputPath(job, new Path(input));
        TextOutputFormat.setOutputPath(job, new Path(output));
        job.setMapperClass(HITS.HITSMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(FSContainer.class);
        job.setReducerClass(HITS.HITSReduser.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(FSContainer.class);
        return job;
    }


    @Override
    public int run(String[] args) throws Exception {
        String input = args[0];
        String out_dir = args[1];

        Integer iterationCount = 5;
        ControlledJob []steps = new ControlledJob[5];

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
