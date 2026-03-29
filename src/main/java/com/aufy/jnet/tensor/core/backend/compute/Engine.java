package com.aufy.jnet.tensor.core.backend.compute;

import com.aufy.jnet.tensor.core.backend.util.ArrayOps;

public class Engine {
  /*
  all operations should create a new tensor and morph the data directly. this engine might be reworked for optimization later
  to save space and not cause the JVM to explode with new objects.

  Engine is by far the closest thing to C or C++ with the amount of manual memory management, so keep a good eye on this one
  because actually understanding what is going on here is very important. unlike C or C++, memory leaks are not an issue because
  all tensor data is in one double[] that is stored neatly.
  */
  
  // broadcasting only works for tensors of the same rank
  public static double[] broadcast(double[] data, int[] originalShape, int[] targetShape) {
    int targetSize = ArrayOps.prod(targetShape);
    int[] dataStrides = PointerLogic.calculateStrides(originalShape);
    int[] targetStrides = PointerLogic.calculateStrides(targetShape);
    
    double[] out = new double[targetSize];

    for (int i = 0; i < targetSize; i++) {
      int originalIndex = 0;
      int remaining = i;

      // map over all the shapes in the targetshape. this keeps parsing the shapes for non 1-dim axes out of the original
      // shape and modify the index to the orginal to pass. virtually, the pointer to the original stays while the transfer
      // pointer into out[] keeps moving.
      for (int dim = 0; dim < targetShape.length; dim++) {
        int coord = remaining / targetStrides[dim];
        remaining %= targetStrides[dim];

        // if its the dimension with a size of 1, dont move the copyer the index
        if (originalShape[dim] != 1) {
          originalIndex += coord * dataStrides[dim];
        }
      }

      out[i] = data[originalIndex];
    }
    return out;
  }

  public static double[] contract(double[] dataA, int[] stridesA, double[] dataB, int[] stridesB, int[] shapeA, int[] axesA, int[] shapeB, int[] axesB, int[] resShape) {
    double[] resData = new double[ArrayOps.prod(resShape)];
    
    int[] subShape = Shaping.getSubShape(shapeA, axesA);
    int contractVolume = ArrayOps.prod(subShape);
    
    // speed thigs up by precomputing the offsets
    int[] OffsetA = Shaping.findOffset(subShape, axesA, stridesA);
    int[] OffsetB = Shaping.findOffset(subShape, axesB, stridesB);
    
    int[] resCoords = new int[resShape.length];
    int resIdx = 0;

    do {
      int baseOffsetA = PointerLogic.mapToOffset(resCoords, new int[axesA.length], axesA, shapeA, stridesA, true);
      int baseOffsetB = PointerLogic.mapToOffset(resCoords, new int[axesB.length], axesB, shapeB, stridesB, false);

      double sum = 0;

      for (int k = 0; k < contractVolume; k++) {
        sum += dataA[baseOffsetA + OffsetA[k]] * dataB[baseOffsetB + OffsetB[k]];
      }
      resData[resIdx++] = sum;

    } while (PointerLogic.nextCoordinate(resCoords, resShape)); // keep transfering while the index is still valid inside the new shape

    return resData;
  }

  public static double[] reduceSum(double[] data, int[] originalShape, int[] targetShape) {
    double[] reduced = new double[ArrayOps.prod(targetShape)];
    int[] gradStrides = PointerLogic.calculateStrides(originalShape);
    int[] origStrides = PointerLogic.calculateStrides(targetShape);

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
      reduced[originalIndex] += data[i];
    }
    return reduced;
  }

  public static double[] concat(int axis, int[][] shapes, double[][] dataList, int[] resultShape) {
    double[] out = new double[ArrayOps.prod(resultShape)];
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

  public static double[] stack(int axis, double[][] dataList, int[] resultShape) {
    // in contiguous models, stacking N tensors of size M is like a single concat
    double[] out = new double[ArrayOps.prod(resultShape)];
    int tensorSize = out.length / dataList.length;
    
    for (int i = 0; i < dataList.length; i++) {
      System.arraycopy(dataList[i], 0, out, i * tensorSize, tensorSize);
    }

    // if axis is not 0, block copy only works if the final result is reshape later in the tensors after the operation. 
    // for a true stack at any axis, use the concat logic with reshaped inputs.
    return out; 
  }

  public static double[] slice(double[] data, int[] shape, int[] strides, int axis, int index) {
    int[] newShape = new int[shape.length - 1];
    for (int i = 0, k = 0; i < shape.length; i++) {
      if (i != axis) newShape[k++] = shape[i];
    }

    double[] out = new double[ArrayOps.prod(newShape)];
    
    int sliceStride = strides[axis];
    // how many times the sliced dimension repeats in the heap
    int iterations = (axis == 0) ? 1 : ArrayOps.prod(shape) / (shape[axis] * sliceStride);
    
    int currentPosition = 0;
    for (int i = 0; i < iterations; i++) {
      // move to the outer block, then jump to the specific slice index
      int originalPosition = (i * shape[axis] * sliceStride) + (index * sliceStride);
      
      System.arraycopy(data, originalPosition, out, currentPosition, sliceStride);
      currentPosition += sliceStride;
    }
    return out;
  }

  public static double[] rangedSlice(double[] data, int[] shape, int[] strides, int axis, int start, int end) {
    int dimensionRange = end - start;
    int[] newShape = shape.clone();
    newShape[axis] = dimensionRange;
    
    double[] out = new double[ArrayOps.prod(newShape)];
    
    int sliceStride = strides[axis];
    int iterations = (axis == 0) ? 1 : ArrayOps.prod(shape) / (shape[axis] * sliceStride);
    
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

  public static double[] place(double[] data, int[] destShape, int[] destStrides, int axis, int index) {
    double[] out = new double[ArrayOps.prod(destShape)];
    int sliceStride = destStrides[axis];
    int iterations = (axis == 0) ? 1 : ArrayOps.prod(destShape) / (destShape[axis] * sliceStride);
    
    int originalPosition = 0;
    for (int i = 0; i < iterations; i++) {
      int finalPosition = (i * destShape[axis] * sliceStride) + (index * sliceStride);
      
      System.arraycopy(data, originalPosition, out, finalPosition, sliceStride);
      originalPosition += sliceStride;
    }

    return out;
  }
}
