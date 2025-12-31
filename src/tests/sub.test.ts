import { describe, it, expect } from "vitest";
import { Tensor } from "../tensor";

describe("Tensor.sub", () => {
  it("subtracts two scalars", () => {
    const a = new Tensor(10);
    const b = new Tensor(3);
    const result = a.sub(b);
    expect(result.data).toEqual(7);
    expect(result.shape).toEqual([]);
  });

  it("subtracts two 1D tensors of same shape", () => {
    const a = new Tensor([10, 20, 30]);
    const b = new Tensor([1, 2, 3]);
    const result = a.sub(b);
    expect(result.data).toEqual([9, 18, 27]);
    expect(result.shape).toEqual([3]);
  });

  it("subtracts two 2D tensors of same shape", () => {
    const a = new Tensor([
      [10, 20],
      [30, 40],
    ]);
    const b = new Tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = a.sub(b);
    expect(result.data).toEqual([
      [9, 18],
      [27, 36],
    ]);
    expect(result.shape).toEqual([2, 2]);
  });

  it("broadcasts scalar to 1D tensor", () => {
    const a = new Tensor([10, 20, 30]);
    const b = new Tensor(5);
    const result = a.sub(b);
    expect(result.data).toEqual([5, 15, 25]);
    expect(result.shape).toEqual([3]);
  });

  it("broadcasts 1D tensor across rows of 2D tensor", () => {
    const a = new Tensor([
      [10, 20, 30],
      [40, 50, 60],
    ]);
    const b = new Tensor([1, 2, 3]);
    const result = a.sub(b);
    expect(result.data).toEqual([
      [9, 18, 27],
      [39, 48, 57],
    ]);
    expect(result.shape).toEqual([2, 3]);
  });

  it("broadcasts column vector to 2D tensor", () => {
    const a = new Tensor([
      [10, 20, 30],
      [40, 50, 60],
    ]);
    const b = new Tensor([[1], [10]]);
    const result = a.sub(b);
    expect(result.data).toEqual([
      [9, 19, 29],
      [30, 40, 50],
    ]);
    expect(result.shape).toEqual([2, 3]);
  });

  it("throws error for incompatible shapes", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor([1, 2]);
    expect(() => a.sub(b)).toThrow("Tensors are incompatible for broadcasting");
  });

  // Edge cases with Infinity and NaN
  it("handles Infinity in subtraction", () => {
    const a = new Tensor([Infinity, -Infinity, 1]);
    const b = new Tensor([1, 1, Infinity]);
    const result = a.sub(b);
    expect(result.data).toEqual([Infinity, -Infinity, -Infinity]);
  });

  it("returns NaN when subtracting Infinity from Infinity", () => {
    const a = new Tensor([Infinity, -Infinity]);
    const b = new Tensor([Infinity, -Infinity]);
    const result = a.sub(b);
    expect(result.data).toEqual([NaN, NaN]);
  });

  it("propagates NaN through subtraction", () => {
    const a = new Tensor([1, NaN, 3]);
    const b = new Tensor([1, 1, NaN]);
    const result = a.sub(b);
    expect(result.data[0]).toEqual(0);
    expect(result.data[1]).toBeNaN();
    expect(result.data[2]).toBeNaN();
  });
});
