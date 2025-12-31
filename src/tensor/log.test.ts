import { describe, it, expect } from "vitest";
import { Tensor } from "./tensor";

describe("Tensor.log", () => {
  it("computes log of a scalar", () => {
    const a = new Tensor(1);
    const result = a.log();
    expect(result.data).toEqual(0); // ln(1) = 0
    expect(result.shape).toEqual([]);
  });

  it("computes log of e", () => {
    const a = new Tensor(Math.E);
    const result = a.log();
    expect(result.data).toBeCloseTo(1); // ln(e) = 1
  });

  it("computes log of a 1D tensor", () => {
    const a = new Tensor([1, Math.E, Math.E ** 2]);
    const result = a.log();
    const data = result.data as number[];
    expect(data[0]).toBeCloseTo(0);
    expect(data[1]).toBeCloseTo(1);
    expect(data[2]).toBeCloseTo(2);
  });

  it("computes log of a 2D tensor", () => {
    const a = new Tensor([
      [1, Math.E],
      [Math.E ** 2, Math.E ** 3],
    ]);
    const result = a.log();
    expect((result.data as number[][])[0][0]).toBeCloseTo(0);
    expect((result.data as number[][])[0][1]).toBeCloseTo(1);
    expect((result.data as number[][])[1][0]).toBeCloseTo(2);
    expect((result.data as number[][])[1][1]).toBeCloseTo(3);
  });

  it("returns -Infinity for zero", () => {
    const a = new Tensor([0]);
    const result = a.log();
    expect(result.data).toEqual([-Infinity]);
  });

  it("returns NaN for negative values", () => {
    const a = new Tensor([-1, -2, -100]);
    const result = a.log();
    const data = result.data as number[];
    expect(data[0]).toBeNaN();
    expect(data[1]).toBeNaN();
    expect(data[2]).toBeNaN();
  });

  it("returns Infinity for Infinity input", () => {
    const a = new Tensor([Infinity]);
    const result = a.log();
    expect(result.data).toEqual([Infinity]);
  });

  it("returns NaN for -Infinity input", () => {
    const a = new Tensor([-Infinity]);
    const result = a.log();
    const data = result.data as number[];
    expect(data[0]).toBeNaN();
  });

  it("propagates NaN", () => {
    const a = new Tensor([1, NaN]);
    const result = a.log();
    const data = result.data as number[];
    expect(data[0]).toEqual(0);
    expect(data[1]).toBeNaN();
  });
});
