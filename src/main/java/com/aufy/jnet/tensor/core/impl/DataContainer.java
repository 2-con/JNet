package com.aufy.jnet.tensor.core.impl;
import java.util.Arrays;

import com.aufy.jnet.tensor.core.backend.compute.PointerLogic;
import com.aufy.jnet.tensor.core.backend.util.ArrayOps;

public class DataContainer {
  // unsafe, but its the engine room so whatever, just don't forget to make all of this mess package-private
  public final double[] data;
  public final int[] shape;
  public final int rank;

  public final int size;
  public final int[] strides;
  
  public DataContainer(double[] data, int... shape) {
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

  public DataContainer(double[] data, int[] shape, int[] strides) {
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

  /**
   * Returns a deep copy of the raw memory buffer of this tensor.
   * 
   * @return a deep copy of the underlying flat array of this tensor
   */
  public double[] dump() {
    return this.data.clone();
  }

  /**
   * Returns the raw memory buffer of this tensor. Unlike .dump(), this does not make a copy and is susceptible to data corruption if managed poorly!
   * 
   * @return the underlying flat array of this tensor
   */
  public double[] rawData() {
    return this.data; // unsafe, but whatever
  }

  public double get(int... indices) {
    if (indices.length != this.rank) {
      throw new IllegalArgumentException("Invalid number of indices.");
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