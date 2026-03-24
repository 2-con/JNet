package tensor.ops;
import tensor.core.Memory;
import tensor.core.Utility;
import tensor.tools.Statistics;

@FunctionalInterface
public interface Reduction {
  double reduce(double[] accumulator);
  
  public static double[] apply(double[] data, int[] shape, int[] strides, int[] axes, Reduction operation, boolean keepDims) {
    int[] resShape = Utility.getSurvivors(shape, axes);
    double[] resData = new double[Statistics.prod(resShape)];
    
    int[] subShape = Utility.getSubShape(shape, axes); // size of one slice
    int reductionVolume = Statistics.prod(subShape);
    
    int[] resCoords = new int[resShape.length];
    int resIdx = 0;
    
    double[] subShapeData = new double[Statistics.prod(subShape)];

    if (resShape.length == 0) {
      for (int k = 0; k < reductionVolume; k++) {
          int[] kCoords = Utility.unravel(k, subShape);
          // use mapToOffset even for scalars to account for strides/offsets
          int offset = Memory.mapToOffset(new int[0], kCoords, axes, shape, strides, true);
          subShapeData[k] = data[offset];
      }
      return new double[]{ operation.reduce(subShapeData) };
    }

    do {
      for (int k = 0; k < reductionVolume; k++) { // fills up the resulting spot by looping over the slice
        int[] kCoords = Utility.unravel(k, subShape); // get the sliced indices of the kth element

        // map coords to offset coors suitable for the 1d data
        int offset = Memory.mapToOffset(resCoords, kCoords, axes, shape, strides, true);
        subShapeData[k] = data[offset];
      }

      double ans = operation.reduce(subShapeData);

      resData[resIdx++] = ans;
    } while (Memory.nextCoordinate(resCoords, resShape));

    return resData;
  }
}
