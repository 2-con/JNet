package com.aufy.jnet.tensor.core.backend.compute;

import com.aufy.jnet.stats.primitive.Statistics;

public class Engine {
  public static double[] transformData(double[] gradOutput, int[] gradShape, int[] inputShape) {
    double[] result = new double[Statistics.prod(inputShape)];
    int[] inputStrides = PointerLogic.calculateStrides(inputShape);
    int[] gradStrides  = PointerLogic.calculateStrides(gradShape);

    for (int i = 0; i < result.length; i++) {
      int remaining = i;
      int gradIndex = 0;

      for (int dim = 0; dim < inputShape.length; dim++) {
        int coord = (remaining / inputStrides[dim]) % inputShape[dim];
        
        if (gradShape[dim] != 1) {
          gradIndex += coord * gradStrides[dim];
        }
      }
      result[i] = gradOutput[gradIndex];
    }
    return result;
  }

  public static double[] broadcast(double[] data, int[] srcShape, int[] targetShape) {
    int targetSize = Statistics.prod(targetShape);
    int[] srcStrides = PointerLogic.calculateStrides(srcShape);
    int[] targetStrides = PointerLogic.calculateStrides(targetShape);
    
    double[] out = new double[targetSize];

    for (int i = 0; i < targetSize; i++) {
      int srcIdx = 0;
      int remaining = i;

      for (int dim = 0; dim < targetShape.length; dim++) {
        int coord = remaining / targetStrides[dim];
        remaining %= targetStrides[dim];

        if (srcShape[dim] != 1) {
          srcIdx += coord * srcStrides[dim];
        }
      }
      out[i] = data[srcIdx];
    }
    return out;
  }

  public static double[] contract(double[] dataA, int[] stridesA, double[] dataB, int[] stridesB, int[] shapeA, int[] axesA, int[] shapeB, int[] axesB, int[] resShape) {
    double[] resData = new double[Statistics.prod(resShape)];
    
    int[] subShape = Shaping.getSubShape(shapeA, axesA);
    int contractVolume = Statistics.prod(subShape);
    
    int[] kOffsetsA = Shaping.precomputeKOffsets(subShape, axesA, stridesA);
    int[] kOffsetsB = Shaping.precomputeKOffsets(subShape, axesB, stridesB);
    
    int[] resCoords = new int[resShape.length];
    int resIdx = 0;
    do {
      int baseOffsetA = PointerLogic.mapToOffset(resCoords, new int[axesA.length], axesA, shapeA, stridesA, true);
      int baseOffsetB = PointerLogic.mapToOffset(resCoords, new int[axesB.length], axesB, shapeB, stridesB, false);

      double sum = 0;

      for (int k = 0; k < contractVolume; k++) {
        sum += dataA[baseOffsetA + kOffsetsA[k]] * dataB[baseOffsetB + kOffsetsB[k]];
      }
      resData[resIdx++] = sum;

    } while (PointerLogic.nextCoordinate(resCoords, resShape));

    return resData;
  }

  public static double[] reduceSum(double[] gradData, int[] gradShape, int[] origShape) {
    double[] reduced = new double[Statistics.prod(origShape)];
    int[] gradStrides = PointerLogic.calculateStrides(gradShape);
    int[] origStrides = PointerLogic.calculateStrides(origShape);

    for (int i = 0; i < gradData.length; i++) { // map over all grad data and map grads by reducing to the new shape
      int remaining = i;
      int origIdx = 0;

      for (int dim = 0; dim < gradShape.length; dim++) {
        int coord = (remaining / gradStrides[dim]) % gradShape[dim];
        
        if (origShape[dim] != 1) {
          origIdx += coord * origStrides[dim];
        }
      }
      reduced[origIdx] += gradData[i];
    }
    return reduced;
  }

  public static double[] concat(int axis, int[][] shapes, double[][] dataList, int[] resultShape) {
    double[] out = new double[Statistics.prod(resultShape)];
    
    int totalAxisDim = resultShape[axis];
    int outerCount = 1;
    for (int i = 0; i < axis; i++) outerCount *= resultShape[i];
    
    int innerSize = 1;
    for (int i = axis + 1; i < resultShape.length; i++) innerSize *= resultShape[i];

    int currentAxisOffset = 0;
    for (int i = 0; i < dataList.length; i++) {
      double[] srcData = dataList[i];
      int srcAxisDim = shapes[i][axis];

      for (int j = 0; j < outerCount; j++) {
        // destination: jump over outer blocks, then jump to the current tensor's start in the axis
        int destPos = (j * totalAxisDim * innerSize) + (currentAxisOffset * innerSize);
        int srcPos = j * srcAxisDim * innerSize;

        System.arraycopy(srcData, srcPos, out, destPos, srcAxisDim * innerSize);
      }
      currentAxisOffset += srcAxisDim;
    }
    return out;
  }

  public static double[] stack(int axis, double[][] dataList, int[] resultShape) {
    // In a contiguous model, stacking N tensors of size M is 
    // identical to a single concat or a bulk copy if the inputs 
    // are already contiguous.
    double[] out = new double[Statistics.prod(resultShape)];
    int tensorSize = out.length / dataList.length;
    
    for (int i = 0; i < dataList.length; i++) {
      System.arraycopy(dataList[i], 0, out, i * tensorSize, tensorSize);
    }

    // Note: If axis != 0, the simple block copy above only works if you reshape the result AFTER. For a true interleaved stack at 
    // any axis, use the concat logic with reshaped inputs.
    return out; 
  }

  public static double[] slice(double[] data, int[] shape, int[] strides, int axis, int sliceIdx) {
    int[] newShape = new int[shape.length - 1];
    for (int i = 0, k = 0; i < shape.length; i++) {
      if (i != axis) newShape[k++] = shape[i];
    }

    double[] out = new double[Statistics.prod(newShape)];
    
    int sliceStride = strides[axis];
    // how many times the sliced dimension repeats in the heap
    int outerIterations = (axis == 0) ? 1 : Statistics.prod(shape) / (shape[axis] * sliceStride);
    
    int currentPos = 0;
    for (int i = 0; i < outerIterations; i++) {
      // move to the outer block, then jump to the specific slice index
      int srcPos = (i * shape[axis] * sliceStride) + (sliceIdx * sliceStride);
      
      System.arraycopy(data, srcPos, out, currentPos, sliceStride);
      currentPos += sliceStride;
    }
    return out;
  }

  public static double[] rangedSlice(double[] data, int[] shape, int[] strides, int axis, int start, int end) {
    int rangeDim = end - start;
    int[] newShape = shape.clone();
    newShape[axis] = rangeDim;
    
    double[] out = new double[Statistics.prod(newShape)];
    
    int sliceStride = strides[axis];
    int outerIterations = (axis == 0) ? 1 : Statistics.prod(shape) / (shape[axis] * sliceStride);
    
    int currentPos = 0;
    for (int i = 0; i < outerIterations; i++) {
      // Jump to outer block, then jump to the start of the range
      int srcPos = (i * shape[axis] * sliceStride) + (start * sliceStride);
      int length = rangeDim * sliceStride;
      
      System.arraycopy(data, srcPos, out, currentPos, length);
      currentPos += length;
    }
    return out;
  }

  public static void place(double[] src, double[] dest, int[] destShape, int[] destStrides, int axis, int index) {
    int sliceStride = destStrides[axis];
    int outerIterations = (axis == 0) ? 1 : Statistics.prod(destShape) / (destShape[axis] * sliceStride);
    
    int srcPos = 0;
    for (int i = 0; i < outerIterations; i++) {
      int destPos = (i * destShape[axis] * sliceStride) + (index * sliceStride);
      
      System.arraycopy(src, srcPos, dest, destPos, sliceStride);
      srcPos += sliceStride;
    }
  }
}
