package com.aufy.jnet.statistics.distributions;

import java.util.Random;

import com.aufy.jnet.core.backend.exceptions.statistics.Parameter;

/**
 * Gaussian distribution.
 */
public class Gaussian extends Distribution {
  private static final Random RNG = new Random();

  private final double mean;
  private final double std;

  public Gaussian(double mean, double std) {
    Parameter.isPositive(std);

    this.mean = mean;
    this.std = std;
  }

  @Override
  public double sample() {
    return mean + std * RNG.nextGaussian();
  }

  @Override
  public double mean() {
    return mean;
  }

  @Override
  public double variance() {
    return std * std;
  }

  @Override
  public String toString() {
    return "Gaussian(mean=" + mean + ", std=" + std + ")";
  }
}
