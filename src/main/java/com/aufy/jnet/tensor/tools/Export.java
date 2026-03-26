package com.aufy.jnet.tensor.tools;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import com.aufy.jnet.tensor.core.impl.CoreTensor;

public class Export {
  public static void saveBinary(CoreTensor tensor, String path) {
    try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(path))) {
        int[] shape = tensor.shape;
        double[] data = tensor.dump();

        dos.writeInt(shape.length); // Save Rank
        for (int dim : shape) dos.writeInt(dim); // Save Dimensions
        for (double val : data) dos.writeDouble(val); // Save Values
        
    } catch (IOException e) {
        System.err.println("Export failed: " + e.getMessage());
    }
  }

  public static CoreTensor loadBinary(String path) {
    try (DataInputStream dis = new DataInputStream(new FileInputStream(path))) {
      int rank = dis.readInt();
      int[] shape = new int[rank];
      int size = 1;
      for (int i = 0; i < rank; i++) {
        shape[i] = dis.readInt();
        size *= shape[i];
      }

      double[] data = new double[size];
      for (int i = 0; i < size; i++) {
        data[i] = dis.readDouble();
      }

      return new CoreTensor(data, shape);
    } catch (IOException e) {
      throw new RuntimeException("Import failed: " + e.getMessage());
    }
  }
}
