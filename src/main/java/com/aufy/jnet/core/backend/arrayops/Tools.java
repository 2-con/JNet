package com.aufy.jnet.core.backend.arrayops;

import java.util.ArrayList;
import java.util.List;

/**
 * Manages raw data processing for both integer and double arrays.
 * 
 * <p>
 * This class provide simple bare-bones operations that modify arrays not based on mathematics. This class contains methods for array processing
 * over mathematical tools; for that, see {@link com.aufy.jnet.core.backend.arrayops.Reductions}
 * <p>
 * 
 */
public class Tools {
  
  /**
   * Reverses an array.
   * 
   * @param array array.
   * @return reversed array.
   */
  public static double[] reverse(double[] array) {
    double[] output = new double[array.length];
    for (int i = 0; i < array.length; i++) {
      output[i] = array[array.length - 1 - i];
    }
    return output;
  }

  /**
   * Reverses an array.
   * 
   * @param array array.
   * @return reversed array.
   */
  public static int[] reverse(int[] array) {
    int[] output = new int[array.length];
    for (int i = 0; i < array.length; i++) {
      output[i] = array[array.length - 1 - i];
    }
    return output;
  }

  /**
   * Removes an element at a specific index from a double array.
   * Elements after the index are shifted to the left.
   *
   * @param array the original double array.
   * @param index the index of the element to remove.
   * @return a new double array with the specified element removed.
   */
  public static double[] remove(double[] array, int index) {
    double[] output = new double[array.length - 1];
    for (int i = 0; i < array.length - 1; i++) {
      if (i < index) output[i] = array[i];
      else output[i] = array[i + 1];
    }
    return output;
  }

  /**
   * Removes an element at a specific index from an int array.
   * Elements after the index are shifted to the left.
   *
   * @param array the original int array.
   * @param index the index of the element to remove.
   * @return a new int array with the specified element removed.
   */
  public static int[] remove(int[] array, int index) {
    int[] output = new int[array.length - 1];
    for (int i = 0; i < array.length - 1; i++) {
      if (i < index) output[i] = array[i];
      else output[i] = array[i + 1];
    }
    return output;
  }

  /**
   * Removes multiple elements from a double array based on a list of indices.
   *
   * @param array the original double array.
   * @param indices the array of indices to be removed.
   * @return a new double array excluding the elements at the specified indices.
   */
  public static double[] remove(double[] array, int[] indices) {
    double[] output = new double[array.length - indices.length];
    int targetIndex = 0;

    for (int i = 0; i < array.length; i++) {
      if (!contains(indices, i)) {
        output[targetIndex] = array[i];
        targetIndex++;
      }
    }
    return output;
  }

  /**
   * Removes multiple elements from an int array based on a list of indices.
   *
   * @param array the original int array.
   * @param indices the array of indices to be removed.
   * @return a new int array excluding the elements at the specified indices.
   */
  public static int[] remove(int[] array, int[] indices) {
    int[] output = new int[array.length - indices.length];
    int targetIndex = 0;

    for (int i = 0; i < array.length; i++) {
      if (!contains(indices, i)) {
        output[targetIndex] = array[i];
        targetIndex++;
      }
    }
    return output;
  }

  /**
   * Checks wether a specific integer is in an array.
   * 
   * @param array array.
   * @param value double to search for.
   * @return boolean if the double is inside the array.
   */
  public static boolean contains(double[] array, double value) {
    for (double i : array) if (i == value) return true;
    return false;
  }

  /**
   * Checks wether a specific integer is in an array.
   * 
   * @param array array.
   * @param value integer to search for.
   * @return boolean if the integer is inside the array.
   */
  public static boolean contains(int[] array, int value) {
    for (int i : array) if (i == value) return true;
    return false;
  }
  
  /**
   * Counts how many a specific integer is in an array.
   * 
   * @param array array.
   * @param value double to search for.
   * @return amount of times the integer is in the array. Returns 0 if not found.
   */
  public static int countContains(double[] array, double value) {
    int count = 0;
    for (double i : array) if (i == value) count++;
    return count;
  }
  
  /**
   * Counts how many a specific integer is in an array.
   * 
   * @param array array.
   * @param value integer to search for.
   * @return amount of times the integer is in the array. Returns 0 if not found.
   */
  public static int countContains(int[] array, int value) {
    int count = 0;
    for (int i : array) if (i == value) count++;
    return count;
  }

  /**
   * Concatenates multiple arrays into one, ordered sequentially.
   * 
   * @param arrays arrays.
   * @return the concatenated array.
   */
  public static double[] concat(double[]... arrays) {
    int totalLength = 0;
    for (double[] array : arrays) {
      totalLength += array.length;
    }
    double[] result = new double[totalLength];
    
    int currentPos = 0;
    for (double[] array : arrays) {
      System.arraycopy(array, 0, result, currentPos, array.length);
      currentPos += array.length;
    }
    return result;
  }

  /**
   * Concatenates multiple arrays into one, ordered sequentially.
   * 
   * @param arrays arrays.
   * @return the concatenated array.
   */
  public static int[] concat(int[]... arrays) {
    int totalLength = 0;
    for (int[] array : arrays) {
      totalLength += array.length;
    }
    int[] result = new int[totalLength];
    
    int currentPos = 0;
    for (int[] array : arrays) {
      System.arraycopy(array, 0, result, currentPos, array.length);
      currentPos += array.length;
    }
    return result;
  }

  /**
   * Gets the index of the first occurrence of a specific value in an array.
   * 
   * @param array array.
   * @param value double to search for.
   * @return index of the value. Returns -1 if not found.
   */
  public static int indexOf(double[] array, double value) {
    for (int i = 0; i < array.length; i++) {
      if (array[i] == value) return i;
    }
    return -1;
  }

  /**
   * Gets the index of the first occurrence of a specific value in an array.
   * 
   * @param array array.
   * @param value integer to search for.
   * @return index of the value. Returns -1 if not found.
   */
  public static int indexOf(int[] array, int value) {
    for (int i = 0; i < array.length; i++) {
      if (array[i] == value) return i;
    }
    return -1;
  }

  /**
   * Arranges an array from 0 to 'size - 1' with ints of increasing order.
   *
   * @param size size of the array
   * @return array containing ints from 0 to 'size - 1'
   */
  public static int[] arrange(int size) {
    int[] result = new int[size];
    for (int i = 0; i < size; i++) result[i] = i;
    return result;
  }

  /**
   * Returns the indices of a specific value in an array.
   * 
   * @param array array to search.
   * @param targetValue value to search for.
   * @return indices of the value.
   */
  public static int[] findIndices(int[] array, int targetValue) {
    List<Integer> tempIndices = new ArrayList<>();
    
    for (int i = 0; i < array.length; i++) {
      if (array[i] == targetValue) {
        tempIndices.add(i);
      }
    }
    
    int[] result = new int[tempIndices.size()];
    for (int i = 0; i < tempIndices.size(); i++) {
      result[i] = tempIndices.get(i);
    }
    
    return result;
  }

  /**
   * Returns the indices of a specific value in an array.
   * 
   * @param array array to search.
   * @param targetValue value to search for.
   * @return indices of the value.
   */
  public static int[] findIndices(double[] array, double targetValue) {
    List<Integer> tempIndices = new ArrayList<>();
    
    for (int i = 0; i < array.length; i++) {
      if (array[i] == targetValue) {
        tempIndices.add(i);
      }
    }
    
    int[] result = new int[tempIndices.size()];
    for (int i = 0; i < tempIndices.size(); i++) {
      result[i] = tempIndices.get(i);
    }
    
    return result;
  }

  /**
   * Returns the elements from specific indecies in an array.
   * 
   * @param data the array to grab elements from.
   * @param indices the indices to grab elements from.
   * @return the elements from the indices.
   */
  public static double[] gather(double[] data, int[] indices) {
    double[] result = new double[indices.length];
    for (int i = 0; i < indices.length; i++) result[i] = data[indices[i]];
    return result;
  }

  /**
   * Returns the elements from specific indecies in an array.
   * 
   * @param data the array to grab elements from.
   * @param indices the indices to grab elements from.
   * @return the elements from the indices.
   */
  public static int[] gather(int[] data, int[] indices) {
    int[] result = new int[indices.length];
    for (int i = 0; i < indices.length; i++) result[i] = data[indices[i]];
    return result;
  }

  /**
   * Replaces all instances of a value in an array with a replacement.
   * 
   * @param data the array to replace values in.
   * @param value the value to search for.
   * @param replacement the value to replace with.
   * @return the array with replaced values.
   */
  public static double[] replace(double[] data, double value, double replacement) {    
    boolean findNaN = Double.isNaN(value);
    double[] result = new double[data.length];
    
    for (int i = 0; i < data.length; i++) {
      boolean match = findNaN ? Double.isNaN(data[i]) : Double.compare(data[i], value) == 0;
      
      result[i] = match ? replacement : data[i];
    }
    
    return result;
  }
}
