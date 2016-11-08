import org.apache.hadoop.io.*;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by nina on 07.11.16.
 */
public class FSContainer implements WritableComparable<FSContainer> {
    private Float value;
    private ArrayList<String> array;

    FSContainer() {
        value = null;
        array = null;
    }
    FSContainer(Float _value, ArrayList<String> _array) {
        value = _value;
        array = _array;
    }

    FSContainer(String _value, String _array) {
        String nullString = new String("");
        if (_value.compareTo(nullString) != 0) {
            value = Float.valueOf(_value);
        } else {
            value = null;
        }

        if (_array.compareTo(nullString) != 0) {
            // Удаление скобок [ ]
            _array = _array.substring(1, _array.length() - 1);
            array = new ArrayList<String>(Arrays.asList(_array.split(", ")));
        } else {
            array = null;
        }
    }

    Float getValue() { return value; }
    ArrayList<String> getArray() { return array; }
    void setValue(Float _val) { value = _val; }
    void setArray(ArrayList<String> _ar) { array = _ar; }


    @Override
    public int compareTo(FSContainer o) {
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
            for (String x : array) {
                (new Text(x)).write(dataOutput);
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
        Text text = new Text();

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
                text.readFields(dataInput);
                array.add(text.toString());
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
