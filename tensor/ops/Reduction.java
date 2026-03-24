package tensor.ops;
import tensor.core.Utility;

@FunctionalInterface
public interface Reduction {
  double reduce(double[] accumulator);
  
  public static double[] apply(double[] data, int[] shape, int[] strides, int[] axes, Reduction operation, boolean keepDims) {
    int[] resShape = Utility.getSurvivors(shape, axes, keepDims);
    double[] resData = new double[Utility.sizeOf(resShape)];
    
    int[] subShape = Utility.getSubShape(shape, axes); // size of one slice
    int reductionVolume = Utility.sizeOf(subShape);
    
    int[] resCoords = new int[resShape.length];
    int resIdx = 0;
    
    double[] subShapeData = new double[Utility.sizeOf(subShape)];

    if (resShape.length == 0) {
      for (int k = 0; k < reductionVolume; k++) {
          int[] kCoords = Utility.unravel(k, subShape);
          // Use mapToOffset even for scalars to account for strides/offsets
          int offset = Utility.mapToOffset(new int[0], kCoords, axes, shape, strides, true, keepDims);
          subShapeData[k] = data[offset];
      }
      return new double[]{ operation.reduce(subShapeData) };
    }

    do {
      for (int k = 0; k < reductionVolume; k++) { // fills up the resulting spot by looping over the slice
        int[] kCoords = Utility.unravel(k, subShape); // get the sliced indices of the kth element

        // map coords to offset coors suitable for the 1d data
        int offset = Utility.mapToOffset(resCoords, kCoords, axes, shape, strides, true, keepDims);
        subShapeData[k] = data[offset];
      }

      double ans = operation.reduce(subShapeData);

      resData[resIdx++] = ans;
    } while (Utility.nextCoordinate(resCoords, resShape));

    return resData;
  }
}
