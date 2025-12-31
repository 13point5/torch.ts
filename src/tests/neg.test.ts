import { describe, it, expect } from "vitest";
import { Tensor } from "../tensor";

describe("Tensor.neg", () => {
  it("negates a scalar", () => {
    const a = new Tensor(5);
    const result = a.neg();
    expect(result.data).toEqual(-5);
    expect(result.shape).toEqual([]);
  });

  it("negates a 1D tensor", () => {
    const a = new Tensor([1, -2, 3, -4]);
    const result = a.neg();
    expect(result.data).toEqual([-1, 2, -3, 4]);
    expect(result.shape).toEqual([4]);
  });

  it("negates a 2D tensor", () => {
    const a = new Tensor([
      [1, -2],
      [-3, 4],
    ]);
    const result = a.neg();
    expect(result.data).toEqual([
      [-1, 2],
      [3, -4],
    ]);
    expect(result.shape).toEqual([2, 2]);
  });

  it("negates zero to zero", () => {
    const a = new Tensor([0, -0]);
    const result = a.neg();
    expect(result.data).toEqual([-0, 0]);
  });

  it("handles Infinity", () => {
    const a = new Tensor([Infinity, -Infinity]);
    const result = a.neg();
    expect(result.data).toEqual([-Infinity, Infinity]);
  });

  it("propagates NaN", () => {
    const a = new Tensor([1, NaN, 3]);
    const result = a.neg();
    const data = result.data as number[];
    expect(data[0]).toEqual(-1);
    expect(data[1]).toBeNaN();
    expect(data[2]).toEqual(-3);
  });
});
