import org.apache.hadoop.io.BooleanWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by nina on 07.11.16.
 */
public class NumberContainer implements WritableComparable<NumberContainer> {
    private Float value;
    private ArrayList<Integer> array;

    NumberContainer() {
        value = null;
        array = null;
    }
    NumberContainer(Float _value, ArrayList<Integer> _array) {
        value = _value;
        array = _array;
    }


    @Override
    public int compareTo(NumberContainer o) {
        return 0;
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        if (value != null) {
            (new BooleanWritable(true)).write(dataOutput);
            (new FloatWritable(value)).write(dataOutput);
        } else {
            (new BooleanWritable(false)).write(dataOutput);
        }

        if (array != null) {
            (new BooleanWritable(true)).write(dataOutput);
            (new IntWritable(array.size())).write(dataOutput);
            for (Integer x : array) {
                (new IntWritable(x)).write(dataOutput);
            }
        } else {
            (new BooleanWritable(false)).write(dataOutput);
        }
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        BooleanWritable wbool = new BooleanWritable();
        FloatWritable wfloat = new FloatWritable();
        IntWritable wint  =  new IntWritable();

        // считавание value
        wbool.readFields(dataInput);
        if (wbool.get()) {
            wfloat.readFields(dataInput);
            value = new Float(wfloat.get());
        } else {
            value = null;
        }

        // считывание array
        wbool.readFields(dataInput);
        if (wbool.get()) {
            wint.readFields(dataInput);
            int n = wint.get();
            array = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                wint.readFields(dataInput);
                array.add(new Integer(wint.get()));
            }
        } else {
            array = null;
        }
    }

    @Override
    public String toString() {
        String string = new String();
        if (value != null) {
            string = value.toString();
        }
        string += "\t";
        if (array != null) {
            string += array.toString();
        }
        return string;
    }
}
