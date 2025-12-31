import { describe, it, expect } from "vitest";
import { Tensor } from "./tensor";

describe("Tensor.shape", () => {
  it("returns empty array for scalar", () => {
    const t = new Tensor(42);
    expect(t.shape).toEqual([]);
  });

  it("returns [n] for 1D tensor", () => {
    const t = new Tensor([1, 2, 3, 4, 5]);
    expect(t.shape).toEqual([5]);
  });

  it("returns [rows, cols] for 2D tensor", () => {
    const t = new Tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(t.shape).toEqual([2, 3]);
  });

  it("returns [d1, d2, d3] for 3D tensor", () => {
    const t = new Tensor([
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      [
        [7, 8],
        [9, 10],
        [11, 12],
      ],
    ]);
    expect(t.shape).toEqual([2, 3, 2]);
  });

  it("returns [1] for single-element 1D tensor", () => {
    const t = new Tensor([7]);
    expect(t.shape).toEqual([1]);
  });

  it("returns [1, 1] for single-element 2D tensor", () => {
    const t = new Tensor([[7]]);
    expect(t.shape).toEqual([1, 1]);
  });

  it("returns correct shape for 4D tensor", () => {
    // Shape: [2, 3, 4, 5]
    const t = new Tensor([
      [
        [
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
        ],
        [
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
        ],
        [
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
        ],
      ],
      [
        [
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
        ],
        [
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
        ],
        [
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
        ],
      ],
    ]);
    expect(t.shape).toEqual([2, 3, 4, 5]);
  });
});

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

  // Tests that expose unflatten bug (shape[0] !== shape.length)
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
});
