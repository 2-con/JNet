package com.aufy.jnet.tensor.core.impl;
import java.util.Arrays;

import com.aufy.jnet.tensor.core.backend.compute.PointerLogic;
import com.aufy.jnet.tensor.core.backend.util.ArrayOps;

public class RawTensor {
  /* TODO: make RawTensor a memory allocator and manager for faster processing
  
  currently, Engine creates and allocate a new double[] every operation. although its fine, its better if there is a global pool to reuse memory
  and space. also, engine should also be purely functional without making any new double[]; rawtensor is fully responsible for memory stuff.
  */
  public final double[] data;
  public final int[] shape;
  public final int rank;

  public final int size;
  public final int[] strides;
  
  public RawTensor(double[] data, int... shape) {
    int expectedSize = 1;
    for (int d : shape) expectedSize *= d;

    if (data.length != expectedSize) {
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be shaped to " + Arrays.toString(shape));
    }

    this.data = data.clone();
    this.shape = shape.clone();
    this.strides = PointerLogic.calculateStrides(shape);
    this.rank = this.shape.length;
    this.size = this.data.length;
  }

  public RawTensor(double[] data, int[] shape, int[] strides) {
    int expectedSize = 1;
    for (int d : shape) expectedSize *= d;

    if (data.length != expectedSize) {
      throw new IllegalArgumentException("Tensor of size " + data.length + " cannot be shaped to " + Arrays.toString(shape));
    }

    this.data = data.clone();
    this.shape = shape.clone();
    this.strides = strides.clone(); // this constructor is private anyways, but just to make sure nothing silly happens
    this.rank = this.shape.length;
    this.size = this.data.length;
  }

  // ########################################################################################################### //
  //                                                  UTILITY                                                    //
  // ########################################################################################################### //

  public int[] getShape() {return this.shape.clone();}
  public int getRank() {return this.rank;}
  public int getSize() {return this.size;}
  public int[] getStrides() {return this.strides.clone();}

  public double[] dump() {
    return this.data.clone();
  }

  public double[] rawData() {
    return this.data; // unsafe, but whatever
  }

  public double get(int... indices) {
    if (indices.length != this.rank) {
      throw new IllegalArgumentException("Invalid number of indices");
    }

    return this.data[PointerLogic.getIndex(this.strides, indices)];
  }

  @Override
  public String toString() {
    if (this.data == null || this.shape == null || this.data.length == 0) return "DataContainer[null]";
    
    String prefix = "DataContainer" + Arrays.toString(this.shape) + "(\n";
    String content = ArrayOps.print(this.data, this.shape, this.strides, 0, 0, 2);

    return prefix + content + "\n)";
  }
}