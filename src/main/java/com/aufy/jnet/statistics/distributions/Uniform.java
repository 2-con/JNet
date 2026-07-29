package com.aufy.jnet.statistics.distributions;

import java.util.Random;

import com.aufy.jnet.core.backend.exceptions.statistics.Parameter;

/**
 * Uniform distribution.
 */
public class Uniform extends Distribution {
  private static final Random RNG = new Random();

  private final double lower;
  private final double upper;

  public Uniform(double lower, double upper) {
    Parameter.isAboveValue(upper, lower, "Uniform");

    this.lower = lower;
    this.upper = upper;
  }

  @Override
  public double sample() {
    return lower + (upper - lower) * RNG.nextDouble();
  }

  @Override
  public double mean() {
    return (upper - lower)/2;
  }

  @Override
  public double variance() {
    return (upper - lower) * (upper - lower) / 12;
  }

  @Override
  public String toString() {
    return "Uniform(upper = " + upper + ", lower = " + lower + ")";
  }
}
