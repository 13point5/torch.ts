import { describe, it, expect } from "vitest";
import { Tensor } from "./tensor";

describe("Tensor.add", () => {
  it("adds two scalars", () => {
    const a = new Tensor(3);
    const b = new Tensor(5);
    const result = a.add(b);
    expect(result.data).toEqual(8);
    expect(result.shape).toEqual([]);
  });

  it("adds two 1D tensors of same shape", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor([4, 5, 6]);
    const result = a.add(b);
    expect(result.data).toEqual([5, 7, 9]);
    expect(result.shape).toEqual([3]);
  });

  it("adds two 2D tensors of same shape", () => {
    const a = new Tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = new Tensor([
      [10, 20],
      [30, 40],
    ]);
    const result = a.add(b);
    expect(result.data).toEqual([
      [11, 22],
      [33, 44],
    ]);
    expect(result.shape).toEqual([2, 2]);
  });

  it("broadcasts scalar to 1D tensor", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor(10);
    const result = a.add(b);
    expect(result.data).toEqual([11, 12, 13]);
    expect(result.shape).toEqual([3]);
  });

  it("broadcasts scalar to 2D tensor", () => {
    const a = new Tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = new Tensor(10);
    const result = a.add(b);
    expect(result.data).toEqual([
      [11, 12],
      [13, 14],
    ]);
    expect(result.shape).toEqual([2, 2]);
  });

  it("broadcasts 1D tensor across rows of 2D tensor", () => {
    const a = new Tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = new Tensor([10, 20, 30]);
    const result = a.add(b);
    expect(result.data).toEqual([
      [11, 22, 33],
      [14, 25, 36],
    ]);
    expect(result.shape).toEqual([2, 3]);
  });

  it("broadcasts column vector to 2D tensor", () => {
    const a = new Tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = new Tensor([[10], [20]]);
    const result = a.add(b);
    expect(result.data).toEqual([
      [11, 12, 13],
      [24, 25, 26],
    ]);
    expect(result.shape).toEqual([2, 3]);
  });

  it("throws error for incompatible shapes", () => {
    const a = new Tensor([1, 2, 3]);
    const b = new Tensor([1, 2]);
    expect(() => a.add(b)).toThrow("Tensors are incompatible for broadcasting");
  });

  it("adds 2D tensors where rows > dimensions [3, 2]", () => {
    const a = new Tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const b = new Tensor([
      [10, 20],
      [30, 40],
      [50, 60],
    ]);
    const result = a.add(b);
    expect(result.data).toEqual([
      [11, 22],
      [33, 44],
      [55, 66],
    ]);
    expect(result.shape).toEqual([3, 2]);
  });

  it("adds 2D tensors with shape [4, 3]", () => {
    const a = new Tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ]);
    const b = new Tensor([100, 200, 300]); // broadcast [3] to [4, 3]
    const result = a.add(b);
    expect(result.data).toEqual([
      [101, 202, 303],
      [104, 205, 306],
      [107, 208, 309],
      [110, 211, 312],
    ]);
    expect(result.shape).toEqual([4, 3]);
  });

  it("adds 3D tensors where first dim > num dimensions [4, 2, 3]", () => {
    const a = new Tensor([
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      [
        [7, 8, 9],
        [10, 11, 12],
      ],
      [
        [13, 14, 15],
        [16, 17, 18],
      ],
      [
        [19, 20, 21],
        [22, 23, 24],
      ],
    ]);
    const b = new Tensor(1); // broadcast scalar
    const result = a.add(b);
    expect(result.data).toEqual([
      [
        [2, 3, 4],
        [5, 6, 7],
      ],
      [
        [8, 9, 10],
        [11, 12, 13],
      ],
      [
        [14, 15, 16],
        [17, 18, 19],
      ],
      [
        [20, 21, 22],
        [23, 24, 25],
      ],
    ]);
    expect(result.shape).toEqual([4, 2, 3]);
  });

  // Edge cases with Infinity and NaN
  it("handles Infinity in addition", () => {
    const a = new Tensor([1, Infinity, -Infinity]);
    const b = new Tensor([1, 1, 1]);
    const result = a.add(b);
    expect(result.data).toEqual([2, Infinity, -Infinity]);
  });

  it("returns NaN when adding Infinity and -Infinity", () => {
    const a = new Tensor([Infinity]);
    const b = new Tensor([-Infinity]);
    const result = a.add(b);
    expect(result.data).toEqual([NaN]);
  });

  it("propagates NaN through addition", () => {
    const a = new Tensor([1, 2, NaN]);
    const b = new Tensor([1, 1, 1]);
    const result = a.add(b);
    expect(result.data[0]).toEqual(2);
    expect(result.data[1]).toEqual(3);
    expect(result.data[2]).toBeNaN();
  });
});

