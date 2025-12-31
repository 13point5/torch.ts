import { Tensor } from "./tensor";

// Demo: matmul (matrix multiplication)
console.log("=== Matrix Multiplication ===");
const A = new Tensor([
  [1, 2],
  [3, 4],
]);
const B = new Tensor([
  [5, 6],
  [7, 8],
]);
console.log("A:", A.toString());
console.log("B:", B.toString());
console.log("A @ B:", A.matmul(B).toString());

// Demo: add (with broadcasting)
console.log("\n=== Addition ===");
const x = new Tensor([
  [1, 2, 3],
  [4, 5, 6],
]);
const y = new Tensor([10, 20, 30]);
console.log("x:", x.toString());
console.log("y:", y.toString());
console.log("x + y:", x.add(y).toString());

// Demo: exp (element-wise exponential)
console.log("\n=== Exponential ===");
const z = new Tensor([0, 1, 2]);
console.log("z:", z.toString());
console.log("exp(z):", z.exp().toString());
