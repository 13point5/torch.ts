import { describe, it, expect } from "vitest";
import { Tensor } from "./tensor";

describe("Tensor.div", () => {
  it("divides two scalars", () => {
    const a = new Tensor(15);
    const b = new Tensor(3);
    const result = a.div(b);
    expect(result.data).toEqual(5);
    expect(result.shape).toEqual([]);
  });

  it("divides two 1D tensors of same shape", () => {
    const a = new Tensor([10, 20, 30]);
    const b = new Tensor([2, 4, 5]);
    const result = a.div(b);
    expect(result.data).toEqual([5, 5, 6]);
    expect(result.shape).toEqual([3]);
  });

  it("divides two 2D tensors of same shape", () => {
    const a = new Tensor([
      [10, 20],
      [30, 40],
    ]);
    const b = new Tensor([
      [2, 4],
      [5, 8],
    ]);
    const result = a.div(b);
    expect(result.data).toEqual([
      [5, 5],
      [6, 5],
    ]);
    expect(result.shape).toEqual([2, 2]);
  });

  it("broadcasts scalar to 1D tensor", () => {
    const a = new Tensor([10, 20, 30]);
    const b = new Tensor(10);
    const result = a.div(b);
    expect(result.data).toEqual([1, 2, 3]);
    expect(result.shape).toEqual([3]);
  });

  it("broadcasts 1D tensor across rows of 2D tensor", () => {
    const a = new Tensor([
      [10, 20, 30],
      [40, 50, 60],
    ]);
    const b = new Tensor([10, 10, 10]);
    const result = a.div(b);
    expect(result.data).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(result.shape).toEqual([2, 3]);
  });

  it("broadcasts column vector to 2D tensor", () => {
    const a = new Tensor([
      [10, 20, 30],
      [40, 50, 60],
    ]);
    const b = new Tensor([[10], [10]]);
    const result = a.div(b);
    expect(result.data).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(result.shape).toEqual([2, 3]);
  });

  it("handles floating point division", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor(2);
    const result = a.div(b);
    expect(result.data).toEqual([0.5, 1, 1.5]);
    expect(result.shape).toEqual([3]);
  });

  it("throws error for incompatible shapes", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor([1, 2]);
    expect(() => a.div(b)).toThrow("Tensors are incompatible for broadcasting");
  });

  it("returns Infinity when dividing positive by zero", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor(0);
    const result = a.div(b);
    expect(result.data).toEqual([Infinity, Infinity, Infinity]);
  });

  it("returns -Infinity when dividing negative by zero", () => {
    const a = new Tensor([-1, -2, -3]);
    const b = new Tensor(0);
    const result = a.div(b);
    expect(result.data).toEqual([-Infinity, -Infinity, -Infinity]);
  });

  it("returns NaN when dividing zero by zero", () => {
    const a = new Tensor([0, 0, 0]);
    const b = new Tensor(0);
    const result = a.div(b);
    expect(result.data).toEqual([NaN, NaN, NaN]);
  });

  it("handles mixed division by zero cases", () => {
    const a = new Tensor([1, -1, 0]);
    const b = new Tensor(0);
    const result = a.div(b);
    expect(result.data).toEqual([Infinity, -Infinity, NaN]);
  });
});
