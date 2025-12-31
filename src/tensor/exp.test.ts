import { describe, it, expect } from "vitest";
import { Tensor } from "./tensor";

describe("Tensor.exp", () => {
  it("computes exp of a scalar", () => {
    const a = new Tensor(0);
    const result = a.exp();
    expect(result.data).toEqual(1); // e^0 = 1
    expect(result.shape).toEqual([]);
  });

  it("computes exp of a 1D tensor", () => {
    const a = new Tensor([0, 1, 2]);
    const result = a.exp();
    const data = result.data as number[];
    expect(data[0]).toBeCloseTo(1); // e^0
    expect(data[1]).toBeCloseTo(Math.E); // e^1
    expect(data[2]).toBeCloseTo(Math.E ** 2); // e^2
  });

  it("computes exp of a 2D tensor", () => {
    const a = new Tensor([
      [0, 1],
      [2, 3],
    ]);
    const result = a.exp();
    expect((result.data as number[][])[0][0]).toBeCloseTo(1);
    expect((result.data as number[][])[0][1]).toBeCloseTo(Math.E);
    expect((result.data as number[][])[1][0]).toBeCloseTo(Math.E ** 2);
    expect((result.data as number[][])[1][1]).toBeCloseTo(Math.E ** 3);
  });

  it("handles negative values", () => {
    const a = new Tensor([-1, -2]);
    const result = a.exp();
    const data = result.data as number[];
    expect(data[0]).toBeCloseTo(1 / Math.E);
    expect(data[1]).toBeCloseTo(1 / Math.E ** 2);
  });

  it("returns Infinity for large positive values", () => {
    const a = new Tensor([1000]);
    const result = a.exp();
    expect(result.data).toEqual([Infinity]);
  });

  it("returns 0 for large negative values", () => {
    const a = new Tensor([-1000]);
    const result = a.exp();
    expect(result.data).toEqual([0]);
  });

  it("handles Infinity input", () => {
    const a = new Tensor([Infinity, -Infinity]);
    const result = a.exp();
    expect(result.data).toEqual([Infinity, 0]);
  });

  it("propagates NaN", () => {
    const a = new Tensor([0, NaN]);
    const result = a.exp();
    const data = result.data as number[];
    expect(data[0]).toEqual(1);
    expect(data[1]).toBeNaN();
  });
});
