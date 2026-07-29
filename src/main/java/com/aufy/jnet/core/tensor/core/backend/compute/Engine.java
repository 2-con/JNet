package com.aufy.jnet.core.tensor.core.backend.compute;

import java.util.Arrays;
import java.util.function.BiFunction;

import com.aufy.jnet.core.backend.arrayops.Reductions;

/**
 * The main engine for calculations and tensor operations under the hood. Operates entirely on raw blocks of memory with no safeguards whatsover.
 * Safeguards are moved to the frontend Tensor class to reduce processing early in development.
 * 
 * this class is not meant to be front-facing, so make sure to add safeguards before operating on raw data.
 */
public class Engine {
  /*
  all operations should create a new tensor and morph the data directly. this engine might be reworked for optimization later
  to save space and not cause the JVM to explode with new objects.

  Engine is by far the closest thing to C or C++ with the amount of manual memory management, so keep a good eye on this one
  because actually understanding what is going on here is very important. unlike C or C++, memory leaks are not an issue because
  all tensor data is in one double[] that is stored neatly.
  */
  
  // broadcasting only works for tensors of the same rank
  // its basically stretching items along an axis
  /**
   * Stretches a tensor along singleton axes into a larger shape.
   * 
   * @param data the data to broadcast.
   * @param originalShape the shape of the original tensor.
   * @param targetShape the shape of the target tensor.
   * @return the broadcasted data.
   */
  public static double[] broadcast(double[] data, int[] originalShape, int[] targetShape) {
    int targetSize = Reductions.prod(targetShape);
    int[] dataStrides = Memory.calculateStrides(originalShape);
    int[] targetStrides = Memory.calculateStrides(targetShape);
    
    double[] out = new double[targetSize]; // output data

    for (int i = 0; i < targetSize; i++) { // counter over output memory
      int originalIndex = 0;
      int remaining = i; // remaining is the index in the output

      // map over all the shapes in the targetshape. this keeps parsing the shapes for non 1-dim axes out of the original
      // shape and modify the index to the orginal to pass. virtually, the pointer to the original stays while the transfer
      // pointer into out[] keeps moving.

      for (int dim = 0; dim < targetShape.length; dim++) { // counter over target shape
        int coord = remaining / targetStrides[dim]; // calculate where this dimension would be in the target tensor
        remaining %= targetStrides[dim];

        if (originalShape[dim] != 1) { // if its not a 1-dim axis
          originalIndex += coord * dataStrides[dim];
        }
      } // loop outputs the location where the original data should be

      out[i] = data[originalIndex];
    }

    return out;
  }

  /**
   * Performs a tesor contraction along specified axes. Tensor contaction is basically sum_[indexes](A[indexes] * B[indexes]) = C[leftover dims].
   * This is a more advanced version of matrix multiplication.
   * 
   * @param dataA the data of the first tensor.
   * @param stridesA the strides of the first tensor.
   * @param dataB the data of the second tensor.
   * @param stridesB the strides of the second tensor.
   * @param shapeA the shape of the first tensor.
   * @param axesA the axes of the first tensor.
   * @param shapeB the shape of the second tensor.
   * @param axesB the axes of the second tensor.
   * @param resShape the shape of the result tensor.
   * @return the data of the result tensor.
   */
  public static double[] contract(double[] dataA, int[] stridesA, double[] dataB, int[] stridesB, int[] shapeA, int[] axesA, int[] shapeB, int[] axesB, int[] resShape) {
    double[] resData = new double[Reductions.prod(resShape)];
    
    // extract the subshape according to the axes
    int[] subShape = Shaping.reorder(shapeA, axesA);
    int contractVolume = Reductions.prod(subShape); // final result volume
    
    // speed thigs up by precomputing the offsets
    int[] OffsetA = Shaping.findOffset(subShape, axesA, stridesA);
    int[] OffsetB = Shaping.findOffset(subShape, axesB, stridesB);
    
    int[] resCoords = new int[resShape.length];
    int resIdx = 0;

    // make sure this thing runs at all cost
    // thankfully the flawed while-loop was fixed: contraction was so wrong before
    // should've probably read more closely before messing around in Engine
    do {
      int baseOffsetA = Memory.mapToOffset(resCoords, new int[axesA.length], axesA, shapeA, stridesA, true);
      int baseOffsetB = Memory.mapToOffset(resCoords, new int[axesB.length], axesB, shapeB, stridesB, false);
      
      double sum = 0;
      
      for (int k = 0; k < contractVolume; k++) {

        // check since the output is wrong on the first go
        if ((baseOffsetA + OffsetA[k]) >= dataA.length || (baseOffsetB + OffsetB[k]) >= dataB.length) {
          System.out.println("shapeB   = " + Arrays.toString(shapeB));
          System.out.println("stridesB = " + Arrays.toString(stridesB));
          System.out.println("resCoords= " + Arrays.toString(resCoords));
          System.out.println("baseOffsetB = " + baseOffsetB);

          System.out.println("=============================================================");

          System.out.println("idxA = " + (baseOffsetA + OffsetA[k]) + " / " + dataA.length);
          System.out.println("idxB = " + (baseOffsetB + OffsetB[k]) + " / " + dataB.length);

          System.out.println("baseOffsetA = " + baseOffsetA);
          System.out.println("baseOffsetB = " + baseOffsetB);

          System.out.println("OffsetA[k] = " + OffsetA[k]);
          System.out.println("OffsetB[k] = " + OffsetB[k]);

          System.out.println("k = " + k);

          System.out.println(Arrays.toString(resCoords));
          throw new RuntimeException();
        }

        sum += dataA[baseOffsetA + OffsetA[k]] * dataB[baseOffsetB + OffsetB[k]];
      }
      
      resData[resIdx] = sum;
      resIdx++;
    } while (
      Memory.nextCoordinate(resCoords, resShape)
    );

    return resData;
  }

  /**
   * Sums elements of an original tensor to a smaller target tensor. This is like a reverse of the broadcast function. must use a bifunction
   * to provide an accumilator function to parse the series of elements in one slice.
   * 
   * @param data the data to sum.
   * @param originalShape the shape of the original tensor.
   * @param targetShape the shape of the target tensor, must have the same rank as the original shape but must have smaller in size.
   * @param operator the operator to apply to each element.
   * @return the reduced data.
   *
   */
  public static double[] reduction(double[] data, int[] originalShape, int[] targetShape, BiFunction<Double, Double, Double> operator) {
    double[] reduced = new double[Reductions.prod(targetShape)];
    int[] gradStrides = Memory.calculateStrides(originalShape);
    int[] origStrides = Memory.calculateStrides(targetShape);

    for (int i = 0; i < data.length; i++) { // map over all grad data and map grads by reducing to the new shape
      int remaining = i;
      int originalIndex = 0;

      // same like broadcasting but its the reverse: its basically ReductionOps::sum/CoreReductionOps::sum but for 
      // the heap itself instead of calling backend.func.Reduction
      for (int dim = 0; dim < originalShape.length; dim++) {
        int coord = (remaining / gradStrides[dim]) % originalShape[dim];
        
        // if its the dimension with a size of 1, dont move the copyer the index
        if (targetShape[dim] != 1) {
          originalIndex += coord * origStrides[dim];
        }
      }
      reduced[originalIndex] = operator.apply(reduced[originalIndex], data[i]);
    }
    return reduced;
  }

  /**
   * Concatenates the tensors (of the same rank) along a specific, existing axis. Functions in Engine are unsafe and have no safety, so make sure (1) the axis is
   * an existing axis and that (2) the shapes outside the concat axis match exactly. JNet will not support padding.
   * 
   * @param axis the axis to concatenate along (must be an existing axis).
   * @param shapes the shapes of the tensors to concatenate.
   * @param dataList the data of the tensors to concatenate.
   * @param resultShape the shape of the resulting tensor.
   * @return the data of the resulting tensor.
   */
  public static double[] concat(int axis, int[][] shapes, double[][] dataList, int[] resultShape) {
    double[] out = new double[Reductions.prod(resultShape)];
    int totalAxisDim = resultShape[axis];
    int outerCount = 1;
    int innerSize = 1;

    for (int i = 0; i < axis; i++) outerCount *= resultShape[i];
    for (int i = axis + 1; i < resultShape.length; i++) innerSize *= resultShape[i];

    int currentAxisOffset = 0;
    for (int i = 0; i < dataList.length; i++) {
      double[] originalData = dataList[i];
      int originalAxisDimension = shapes[i][axis];

      for (int j = 0; j < outerCount; j++) {
        // jump over outer blocks, then jump to the current tensor's start in the axis
        int finalPosition = (j * totalAxisDim * innerSize) + (currentAxisOffset * innerSize);
        int originalPosition = j * originalAxisDimension * innerSize;

        System.arraycopy(originalData, originalPosition, out, finalPosition, originalAxisDimension * innerSize);
      }
      currentAxisOffset += originalAxisDimension;
    }
    return out;
  }

  /**
   * Stacks tensors (of the same rank) along a specific, new axis. Functions in Engine are unsafe and have no safety, so make sure every tensor
   * has the same shape. JNet will not support padding.
   * 
   * @param axis the axis to stack along (must be a new axis).
   * @param dataList the data of the tensors to stack.
   * @param resultShape the shape of the resulting tensor.
   * @param originalShapes the shapes of the original tensors.
   * @return the data of the resulting tensor.
   */
  public static double[] stack(int axis, double[][] dataList, int[][] originalShapes, int[] resultShape) {
    // if axis is 0, keep the sequential copy
    if (axis == 0) {
      double[] out = new double[Reductions.prod(resultShape)];
      int tensorSize = out.length / dataList.length;
      for (int i = 0; i < dataList.length; i++) {
        System.arraycopy(dataList[i], 0, out, i * tensorSize, tensorSize);
      }
      return out;
    }

    // if axis > 0, unsqueeze shapes by inserting a 1 at [axis]
    int[][] unsqueezedShapes = new int[originalShapes.length][];

    for (int i = 0; i < originalShapes.length; i++) {
      int[] newShape = new int[unsqueezedShapes[i].length + 1];

      System.arraycopy(unsqueezedShapes[i], 0, newShape, 0, axis); // copy original shape up until axis
      newShape[axis] = 1; // 1 at axis
      System.arraycopy(unsqueezedShapes[i], axis, newShape, axis + 1, unsqueezedShapes[i].length - axis); // everything else after the axis

      unsqueezedShapes[i] = newShape; // this new shape is now replaced
    }

    // delegate to concat
    return concat(axis, unsqueezedShapes, dataList, resultShape);
  }

  /**
   * Extracts a slice of a tensor along a specific axis and at a specific index in that axis. It effectively removes one dimension from the tensor.
   * 
   * @param data the data of the tensor.
   * @param shape the shape of the tensor.
   * @param strides the strides of the tensor.
   * @param axis the axis to slice along.
   * @param index the index to slice at.
   * @return the data of the sliced tensor.
   */
  public static double[] slice(double[] data, int[] shape, int[] strides, int axis, int index) {
    int[] newShape = new int[shape.length - 1];
    for (int i = 0, k = 0; i < shape.length; i++) {
      if (i != axis) newShape[k++] = shape[i];
    }

    double[] out = new double[Reductions.prod(newShape)];
    
    int sliceStride = strides[axis];
    // how many times the sliced dimension repeats in the heap
    int iterations = (axis == 0) ? 1 : Reductions.prod(shape) / (shape[axis] * sliceStride);
    
    int currentPosition = 0;
    for (int i = 0; i < iterations; i++) {
      // move to the outer block, then jump to the specific slice index
      int originalPosition = (i * shape[axis] * sliceStride) + (index * sliceStride);
      
      System.arraycopy(data, originalPosition, out, currentPosition, sliceStride);
      currentPosition += sliceStride;
    }
    return out;
  }

  /**
   * Extracts a slice of a tensor along a specific axis and at a specific index in that axis. Unlike slice, it supports a slice range; it does not
   * naively remove a dimension but keeps it at a singleton (if applicable).
   * 
   * @param data the data of the tensor.
   * @param shape the shape of the tensor.
   * @param strides the strides of the tensor.
   * @param axis the axis to slice along.
   * @param start the start index of the slice range.
   * @param end the end index of the slice range.
   * @return the data of the sliced tensor.
   */
  public static double[] rangedSlice(double[] data, int[] shape, int[] strides, int axis, int start, int end) {
    int dimensionRange = end - start;
    int[] newShape = shape.clone();
    newShape[axis] = dimensionRange;
    
    double[] out = new double[Reductions.prod(newShape)];
    
    int sliceStride = strides[axis];
    int iterations = (axis == 0) ? 1 : Reductions.prod(shape) / (shape[axis] * sliceStride);
    
    int currentPosition = 0;
    for (int i = 0; i < iterations; i++) {
      // jump to outer block, then jump to the start of the range
      int originalPosition = (i * shape[axis] * sliceStride) + (start * sliceStride);
      int length = dimensionRange * sliceStride;
      
      System.arraycopy(data, originalPosition, out, currentPosition, length);
      currentPosition += length;
    }
    return out;
  }

  /**
   * Puts an tensor subshape in the place of a specific axis and at a specific index in that axis.
   * 
   * @param data the data of the tensor.
   * @param destShape the shape of the destination tensor.
   * @param destStrides the strides of the destination tensor.
   * @param axis the axis to slice along.
   * @param index the index to slice at.
   * @return the data of the destination tensor with this tensor stuffed into it.
   */
  public static double[] insert(double[] data, int[] destShape, int[] destStrides, int axis, int index) {
    double[] out = new double[Reductions.prod(destShape)];
    int sliceStride = destStrides[axis];
    int iterations = (axis == 0) ? 1 : Reductions.prod(destShape) / (destShape[axis] * sliceStride);
    
    int originalPosition = 0;
    for (int i = 0; i < iterations; i++) {
      int finalPosition = (i * destShape[axis] * sliceStride) + (index * sliceStride);
      
      System.arraycopy(data, originalPosition, out, finalPosition, sliceStride);
      originalPosition += sliceStride;
    }

    return out;
  }

  /**
   * Rearranges a non-contiguous data array into a contiguous one data array based on a new shape and strides.
   * 
   * @param data the non-contiguous data array.
   * @param shape the current shape of the tensor.
   * @param strides the current non-contiguous strides.
   * @return a new flat array with elements rearranged in standard contiguous order.
   */
  public static double[] makeContiguous(double[] data, int[] shape, int[] strides) {
    int totalElements = Reductions.prod(shape);
    double[] out = new double[totalElements];
    
    for (int i = 0; i < totalElements; i++) {
      int[] logicalCoords = Memory.unravel(i, shape);
      
      int srcIndex = Memory.getIndex(strides, logicalCoords);
      
      out[i] = data[srcIndex];
    }
    
    return out;
  }

}
