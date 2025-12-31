import { describe, it, expect } from "vitest";
import { Tensor } from "../tensor";

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
