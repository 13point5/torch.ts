import { describe, it, expect } from "vitest";
import { Tensor } from "../tensor";
import { IndexOutOfBoundsError, InvalidIndexError } from "../errors";

describe("Tensor.pos", () => {
  describe("valid indices", () => {
    it("returns correct value for 1D tensor", () => {
      const t = new Tensor([10, 20, 30, 40]);
      expect(t.pos([0])).toBe(10);
      expect(t.pos([1])).toBe(20);
      expect(t.pos([3])).toBe(40);
    });

    it("returns correct value for 2D tensor", () => {
      const t = new Tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(t.pos([0, 0])).toBe(1);
      expect(t.pos([0, 2])).toBe(3);
      expect(t.pos([1, 0])).toBe(4);
      expect(t.pos([1, 2])).toBe(6);
    });

    it("returns correct value for 3D tensor", () => {
      const t = new Tensor([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      expect(t.pos([0, 0, 0])).toBe(1);
      expect(t.pos([0, 1, 1])).toBe(4);
      expect(t.pos([1, 0, 0])).toBe(5);
      expect(t.pos([1, 1, 1])).toBe(8);
    });
  });

  describe("InvalidIndexError", () => {
    it("throws when too few indices provided", () => {
      const t = new Tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(() => t.pos([0])).toThrow(InvalidIndexError);
    });

    it("throws when too many indices provided", () => {
      const t = new Tensor([1, 2, 3]);
      expect(() => t.pos([0, 0])).toThrow(InvalidIndexError);
    });

    it("throws when no indices provided for non-scalar", () => {
      const t = new Tensor([1, 2, 3]);
      expect(() => t.pos([])).toThrow(InvalidIndexError);
    });
  });

  describe("IndexOutOfBoundsError", () => {
    it("throws when index is negative", () => {
      const t = new Tensor([1, 2, 3]);
      expect(() => t.pos([-1])).toThrow(IndexOutOfBoundsError);
    });

    it("throws when index equals dimension size", () => {
      const t = new Tensor([1, 2, 3]);
      expect(() => t.pos([3])).toThrow(IndexOutOfBoundsError);
    });

    it("throws when index exceeds dimension size", () => {
      const t = new Tensor([1, 2, 3]);
      expect(() => t.pos([5])).toThrow(IndexOutOfBoundsError);
    });

    it("throws when any index in multi-dim is out of bounds", () => {
      const t = new Tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(() => t.pos([2, 0])).toThrow(IndexOutOfBoundsError);
      expect(() => t.pos([0, 3])).toThrow(IndexOutOfBoundsError);
    });
  });
});
