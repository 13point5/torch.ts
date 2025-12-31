import { describe, it, expect } from "vitest";
import { Tensor } from "../tensor";

describe("Tensor.matmul", () => {
  describe("2D matrix multiplication", () => {
    it("multiplies two 2D matrices [2,3] @ [3,2]", () => {
      const a = new Tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = new Tensor([
        [7, 8],
        [9, 10],
        [11, 12],
      ]);
      const result = a.matmul(b);
      // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
      // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
      expect(result.data).toEqual([
        [58, 64],
        [139, 154],
      ]);
      expect(result.shape).toEqual([2, 2]);
    });

    it("multiplies [3,2] @ [2,4]", () => {
      const a = new Tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const b = new Tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const result = a.matmul(b);
      // Row 0: [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11, 14, 17, 20]
      // Row 1: [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8] = [23, 30, 37, 44]
      // Row 2: [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8] = [35, 46, 57, 68]
      expect(result.data).toEqual([
        [11, 14, 17, 20],
        [23, 30, 37, 44],
        [35, 46, 57, 68],
      ]);
      expect(result.shape).toEqual([3, 4]);
    });

    it("multiplies square matrices [2,2] @ [2,2]", () => {
      const a = new Tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = new Tensor([
        [5, 6],
        [7, 8],
      ]);
      const result = a.matmul(b);
      // [1*5+2*7, 1*6+2*8] = [19, 22]
      // [3*5+4*7, 3*6+4*8] = [43, 50]
      expect(result.data).toEqual([
        [19, 22],
        [43, 50],
      ]);
      expect(result.shape).toEqual([2, 2]);
    });
  });

  describe("1D vector operations", () => {
    it("computes dot product of two 1D vectors [3] @ [3]", () => {
      const a = new Tensor([1, 2, 3]);
      const b = new Tensor([4, 5, 6]);
      const result = a.matmul(b);
      // 1*4 + 2*5 + 3*6 = 32
      expect(result.data).toEqual(32);
      expect(result.shape).toEqual([]);
    });

    it("computes matrix-vector product [2,3] @ [3]", () => {
      const a = new Tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = new Tensor([1, 2, 3]);
      const result = a.matmul(b);
      // [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
      expect(result.data).toEqual([14, 32]);
      expect(result.shape).toEqual([2]);
    });

    it("computes vector-matrix product [3] @ [3,2]", () => {
      const a = new Tensor([1, 2, 3]);
      const b = new Tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const result = a.matmul(b);
      // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
      expect(result.data).toEqual([22, 28]);
      expect(result.shape).toEqual([2]);
    });
  });

  describe("batched matrix multiplication", () => {
    it("multiplies batched 3D tensors [2,3,4] @ [2,4,5]", () => {
      // Batch of 2 matrices
      const a = new Tensor([
        [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
        ],
        [
          [2, 0, 0, 0],
          [0, 2, 0, 0],
          [0, 0, 2, 0],
        ],
      ]);
      const b = new Tensor([
        [
          [1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [11, 12, 13, 14, 15],
          [16, 17, 18, 19, 20],
        ],
        [
          [1, 1, 1, 1, 1],
          [2, 2, 2, 2, 2],
          [3, 3, 3, 3, 3],
          [4, 4, 4, 4, 4],
        ],
      ]);
      const result = a.matmul(b);
      expect(result.shape).toEqual([2, 3, 5]);
      // First batch: identity-ish extracts first 3 rows
      expect((result.data as number[][][])[0]).toEqual([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
      ]);
      // Second batch: 2x identity extracts and doubles first 3 rows
      expect((result.data as number[][][])[1]).toEqual([
        [2, 2, 2, 2, 2],
        [4, 4, 4, 4, 4],
        [6, 6, 6, 6, 6],
      ]);
    });

    it("broadcasts 2D matrix against batched 3D tensor [2,3,4] @ [4,5]", () => {
      const a = new Tensor([
        [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
        ],
        [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
        ],
      ]);
      const b = new Tensor([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
      ]);
      const result = a.matmul(b);
      expect(result.shape).toEqual([2, 3, 5]);
      // Both batches get same result (extracting first 3 rows of b)
      const expected = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
      ];
      expect((result.data as number[][][])[0]).toEqual(expected);
      expect((result.data as number[][][])[1]).toEqual(expected);
    });

    it("broadcasts 2D matrix against batched 3D tensor [3,4] @ [2,4,5]", () => {
      const a = new Tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
      ]);
      const b = new Tensor([
        [
          [1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [11, 12, 13, 14, 15],
          [16, 17, 18, 19, 20],
        ],
        [
          [1, 1, 1, 1, 1],
          [2, 2, 2, 2, 2],
          [3, 3, 3, 3, 3],
          [4, 4, 4, 4, 4],
        ],
      ]);
      const result = a.matmul(b);
      expect(result.shape).toEqual([2, 3, 5]);
    });

    it("multiplies 4D batched tensors [2,3,4,5] @ [2,3,5,6] (multi-head attention style)", () => {
      // Shape: [batch=2, heads=3, seq=4, dim=5] @ [batch=2, heads=3, dim=5, out=6]
      // This is like Q @ K^T in multi-head attention
      // Using simple values: each 2D matrix is filled with 1s
      const a = new Tensor([
        [
          [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
          ],
          [
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
          ],
          [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
          ],
        ],
        [
          [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
          ],
          [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
          ],
          [
            [3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3],
          ],
        ],
      ]);
      const b = new Tensor([
        [
          [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
          ],
          [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
          ],
          [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
          ],
        ],
        [
          [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
          ],
          [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
          ],
          [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
          ],
        ],
      ]);
      const result = a.matmul(b);
      expect(result.shape).toEqual([2, 3, 4, 6]);
      // [1,1,1,1,1] @ [[1]*6, [1]*6, ...] = [5,5,5,5,5,5] for each row
      // For a[0][0] (all 1s), each row sums to 5
      expect((result.data as number[][][][])[0][0][0]).toEqual([
        5, 5, 5, 5, 5, 5,
      ]);
      // For a[0][1] (all 2s), each row sums to 10
      expect((result.data as number[][][][])[0][1][0]).toEqual([
        10, 10, 10, 10, 10, 10,
      ]);
      // For a[1][2] (all 3s), each row sums to 15
      expect((result.data as number[][][][])[1][2][0]).toEqual([
        15, 15, 15, 15, 15, 15,
      ]);
    });

    it("complex batch broadcast [2,1,3,4] @ [3,4,5] -> [2,3,3,5]", () => {
      // A has shape [2, 1, 3, 4] - batch dims [2, 1]
      // B has shape [3, 4, 5] - batch dims [3]
      // Broadcast: [2, 1] with [3] -> [2, 3]
      // Result: [2, 3, 3, 5]
      const a = new Tensor([
        [
          // batch [0, 0] - this will broadcast to [0,0], [0,1], [0,2]
          [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
          ],
        ],
        [
          // batch [1, 0] - this will broadcast to [1,0], [1,1], [1,2]
          [
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
          ],
        ],
      ]);
      const b = new Tensor([
        // batch [0]
        [
          [1, 1, 1, 1, 1],
          [2, 2, 2, 2, 2],
          [3, 3, 3, 3, 3],
          [4, 4, 4, 4, 4],
        ],
        // batch [1]
        [
          [10, 10, 10, 10, 10],
          [20, 20, 20, 20, 20],
          [30, 30, 30, 30, 30],
          [40, 40, 40, 40, 40],
        ],
        // batch [2]
        [
          [100, 100, 100, 100, 100],
          [200, 200, 200, 200, 200],
          [300, 300, 300, 300, 300],
          [400, 400, 400, 400, 400],
        ],
      ]);
      const result = a.matmul(b);
      expect(result.shape).toEqual([2, 3, 3, 5]);

      // result[0, 0] = a[0, 0] @ b[0] = identity-ish @ b[0] = first 3 rows of b[0]
      expect((result.data as number[][][][])[0][0]).toEqual([
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
      ]);

      // result[0, 1] = a[0, 0] @ b[1] (a broadcasts, uses same slice)
      expect((result.data as number[][][][])[0][1]).toEqual([
        [10, 10, 10, 10, 10],
        [20, 20, 20, 20, 20],
        [30, 30, 30, 30, 30],
      ]);

      // result[1, 2] = a[1, 0] @ b[2] = 2x identity-ish @ b[2] = doubled first 3 rows
      expect((result.data as number[][][][])[1][2]).toEqual([
        [200, 200, 200, 200, 200],
        [400, 400, 400, 400, 400],
        [600, 600, 600, 600, 600],
      ]);
    });
  });

  describe("error cases", () => {
    it("throws error when inner dimensions don't match [2,3] @ [4,5]", () => {
      const a = new Tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = new Tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
      ]);
      expect(() => a.matmul(b)).toThrow();
    });

    it("throws error when 1D vectors have different lengths", () => {
      const a = new Tensor([1, 2, 3]);
      const b = new Tensor([1, 2, 3, 4]);
      expect(() => a.matmul(b)).toThrow();
    });

    it("throws error when batch dimensions are incompatible [2,3,4] @ [3,4,5]", () => {
      // Batch dim 2 vs 3 - not broadcastable (neither is 1)
      const a = new Tensor([
        [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
        ],
        [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
        ],
      ]);
      const b = new Tensor([
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
      ]);
      expect(() => a.matmul(b)).toThrow();
    });
  });

  describe("identity and special cases", () => {
    it("multiplying by identity matrix returns same values", () => {
      const a = new Tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const identity = new Tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const result = a.matmul(identity);
      expect(result.data).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it("handles single element matrices", () => {
      const a = new Tensor([[5]]);
      const b = new Tensor([[3]]);
      const result = a.matmul(b);
      expect(result.data).toEqual([[15]]);
      expect(result.shape).toEqual([1, 1]);
    });
  });
});
