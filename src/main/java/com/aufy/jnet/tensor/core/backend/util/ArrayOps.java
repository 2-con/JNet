package com.aufy.jnet.tensor.core.backend.util;

public class ArrayOps {
  /*
  JNet is partially a tensor library, not an array library, so these are some nice things to have for larger operations.
  these are more readable than long lambda funcs. these functions should be mathematical in nature
  */

  public static double sum(double[] array) {
    double ans = 0;
    for (double n : array) ans += n;
    return ans;
  }

  public static double prod(double[] array) {
    double ans = 1;
    for (double n : array) ans *= n;
    return ans;
  }

  public static double min(double[] array) {
    double ans = Double.MAX_VALUE;
    for (double n : array) ans = (n < ans) ? n : ans;
    return ans;
  }

  public static double max(double[] array) {
    double ans = Double.MIN_VALUE;
    for (double n : array) ans = (n > ans) ? n : ans;
    return ans;
  }

  // int arrays

  public static int sum(int[] array) {
    int ans = 0;
    for (int n : array) ans += n;
    return ans;
  }

  public static int prod(int[] array) {
    int ans = 1;
    for (double n : array) ans *= n;
    return ans;
  }

  public static int min(int[] array) {
    int ans = Integer.MAX_VALUE;
    for (int n : array) ans = Math.min(n, ans);
    return ans;
  }

  public static int max(int[] array) {
    int ans = Integer.MIN_VALUE;
    for (int n : array) ans = Math.max(n, ans);
    return ans;
  }
}
