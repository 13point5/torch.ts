import { describe, it, expect } from "vitest";
import { Tensor } from "../tensor";

describe("Tensor.mul", () => {
  it("multiplies two scalars", () => {
    const a = new Tensor(3);
    const b = new Tensor(5);
    const result = a.mul(b);
    expect(result.data).toEqual(15);
    expect(result.shape).toEqual([]);
  });

  it("multiplies two 1D tensors of same shape", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor([4, 5, 6]);
    const result = a.mul(b);
    expect(result.data).toEqual([4, 10, 18]);
    expect(result.shape).toEqual([3]);
  });

  it("multiplies two 2D tensors of same shape", () => {
    const a = new Tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = new Tensor([
      [10, 20],
      [30, 40],
    ]);
    const result = a.mul(b);
    expect(result.data).toEqual([
      [10, 40],
      [90, 160],
    ]);
    expect(result.shape).toEqual([2, 2]);
  });

  it("broadcasts scalar to 1D tensor", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor(10);
    const result = a.mul(b);
    expect(result.data).toEqual([10, 20, 30]);
    expect(result.shape).toEqual([3]);
  });

  it("broadcasts 1D tensor across rows of 2D tensor", () => {
    const a = new Tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = new Tensor([10, 20, 30]);
    const result = a.mul(b);
    expect(result.data).toEqual([
      [10, 40, 90],
      [40, 100, 180],
    ]);
    expect(result.shape).toEqual([2, 3]);
  });

  it("broadcasts column vector to 2D tensor", () => {
    const a = new Tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = new Tensor([[10], [100]]);
    const result = a.mul(b);
    expect(result.data).toEqual([
      [10, 20, 30],
      [400, 500, 600],
    ]);
    expect(result.shape).toEqual([2, 3]);
  });

  it("throws error for incompatible shapes", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor([1, 2]);
    expect(() => a.mul(b)).toThrow("Tensors are incompatible for broadcasting");
  });

  // Edge cases with Infinity and NaN
  it("handles Infinity in multiplication", () => {
    const a = new Tensor([Infinity, -Infinity, 2]);
    const b = new Tensor([2, 2, Infinity]);
    const result = a.mul(b);
    expect(result.data).toEqual([Infinity, -Infinity, Infinity]);
  });

  it("returns NaN when multiplying Infinity by zero", () => {
    const a = new Tensor([Infinity, -Infinity, 0]);
    const b = new Tensor([0, 0, Infinity]);
    const result = a.mul(b);
    expect(result.data).toEqual([NaN, NaN, NaN]);
  });

  it("handles Infinity signs correctly", () => {
    const a = new Tensor([Infinity, Infinity, -Infinity, -Infinity]);
    const b = new Tensor([1, -1, 1, -1]);
    const result = a.mul(b);
    expect(result.data).toEqual([Infinity, -Infinity, -Infinity, Infinity]);
  });

  it("propagates NaN through multiplication", () => {
    const a = new Tensor([1, NaN, 3]);
    const b = new Tensor([2, 2, NaN]);
    const result = a.mul(b);
    expect(result.data[0]).toEqual(2);
    expect(result.data[1]).toBeNaN();
    expect(result.data[2]).toBeNaN();
  });
});
