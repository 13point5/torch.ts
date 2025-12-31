import { describe, it, expect } from "vitest";
import { Tensor } from "../tensor";

describe("Tensor.sqrt", () => {
  it("computes sqrt of a scalar", () => {
    const a = new Tensor(4);
    const result = a.sqrt();
    expect(result.data).toEqual(2);
    expect(result.shape).toEqual([]);
  });

  it("computes sqrt of a 1D tensor", () => {
    const a = new Tensor([0, 1, 4, 9, 16]);
    const result = a.sqrt();
    expect(result.data).toEqual([0, 1, 2, 3, 4]);
  });

  it("computes sqrt of a 2D tensor", () => {
    const a = new Tensor([
      [1, 4],
      [9, 16],
    ]);
    const result = a.sqrt();
    expect(result.data).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it("handles non-perfect squares", () => {
    const a = new Tensor([2, 3, 5]);
    const result = a.sqrt();
    const data = result.data as number[];
    expect(data[0]).toBeCloseTo(Math.sqrt(2));
    expect(data[1]).toBeCloseTo(Math.sqrt(3));
    expect(data[2]).toBeCloseTo(Math.sqrt(5));
  });

  it("returns 0 for zero", () => {
    const a = new Tensor([0]);
    const result = a.sqrt();
    expect(result.data).toEqual([0]);
  });

  it("returns NaN for negative values", () => {
    const a = new Tensor([-1, -4, -9]);
    const result = a.sqrt();
    const data = result.data as number[];
    expect(data[0]).toBeNaN();
    expect(data[1]).toBeNaN();
    expect(data[2]).toBeNaN();
  });

  it("returns Infinity for Infinity input", () => {
    const a = new Tensor([Infinity]);
    const result = a.sqrt();
    expect(result.data).toEqual([Infinity]);
  });

  it("returns NaN for -Infinity input", () => {
    const a = new Tensor([-Infinity]);
    const result = a.sqrt();
    const data = result.data as number[];
    expect(data[0]).toBeNaN();
  });

  it("propagates NaN", () => {
    const a = new Tensor([4, NaN, 9]);
    const result = a.sqrt();
    const data = result.data as number[];
    expect(data[0]).toEqual(2);
    expect(data[1]).toBeNaN();
    expect(data[2]).toEqual(3);
  });
});
